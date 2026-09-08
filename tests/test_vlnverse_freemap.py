"""Unit tests for the Isaac motion layer (``sim/isaac/quat.py`` / ``freemap.py``):
numpy only, no Isaac. Ported from the AgentCanvas ``env_vlnverse`` nodeset tests."""

from __future__ import annotations

import numpy as np
import pytest

from embodiedscore_envs.benchmarks.env.sim.isaac.freemap import FreemapKinematics
from embodiedscore_envs.benchmarks.env.sim.isaac.quat import euler_angles_to_quat, quat_to_euler_angles

# ==================== quaternion helpers ====================


def test_yaw_quat_hand_pinned():
    # euler [0,0,pi/2] -> (w,x,y,z) = (cos(pi/4), 0, 0, sin(pi/4))
    q = euler_angles_to_quat(np.array([0.0, 0.0, np.pi / 2]))
    assert np.allclose(q, [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)], atol=1e-12)


@pytest.mark.parametrize("theta", [-3.0, -np.pi / 2, -0.3, 0.0, 0.7, np.pi / 2, 2.9])
def test_quat_heading_round_trip(theta):
    q = euler_angles_to_quat(np.array([0.0, 0.0, theta]))
    rpy = quat_to_euler_angles(q)
    assert abs(rpy[0]) < 1e-9 and abs(rpy[1]) < 1e-9
    assert np.isclose(rpy[2], theta, atol=1e-9)


def test_quat_parity_with_scipy():
    scipy_transform = pytest.importorskip("scipy.spatial.transform")
    R = scipy_transform.Rotation
    rng = np.random.default_rng(0)
    for _ in range(20):
        rpy = rng.uniform(-1.4, 1.4, size=3)  # stay clear of gimbal lock
        ours = euler_angles_to_quat(rpy)
        s = R.from_euler("xyz", rpy).as_quat()  # (x,y,z,w)
        theirs = np.array([s[3], s[0], s[1], s[2]])
        assert np.allclose(ours, theirs, atol=1e-9) or np.allclose(ours, -theirs, atol=1e-9)
        back = quat_to_euler_angles(theirs)
        theirs_rpy = R.from_quat(s).as_euler("xyz")
        assert np.allclose(back, theirs_rpy, atol=1e-9)


# ==================== movement geometry ====================


def _kin_free() -> FreemapKinematics:
    k = FreemapKinematics(use_occupancy_collision=False)
    k.place(np.array([0.0, 0.0, 1.2]), euler_angles_to_quat(np.array([0.0, 0.0, 0.0])))
    return k


def test_place_sets_heading_from_quaternion():
    k = FreemapKinematics(use_occupancy_collision=False)
    yaw = -1.2073438087183336
    k.collision_occurred = True
    k.place([-2.75, 2.98, 1.2], euler_angles_to_quat(np.array([0.0, 0.0, yaw])))
    assert k.collision_occurred is False
    assert np.allclose(k.agent_position, [-2.75, 2.98, 1.2])
    assert np.isclose(k.agent_heading, yaw, atol=1e-9)


def test_straight_move_advances_along_heading():
    k = _kin_free()
    collision, dev = k.move(0.0, 0.0, 2.0)
    assert collision is False and dev == 0.0
    assert np.allclose(k.agent_position, [2.0, 0.0, 1.2], atol=1e-9)
    assert np.isclose(k.agent_heading, 0.0, atol=1e-9)


def test_heading_composition():
    k = _kin_free()
    k.move(np.pi / 2, 0.0, 1.0)  # face +y, advance 1
    assert np.isclose(k.agent_heading, np.pi / 2, atol=1e-9)
    assert np.allclose(k.agent_position, [0.0, 1.0, 1.2], atol=1e-9)
    k.move(np.pi / 2, 0.0, 1.0)  # face -x (heading pi), advance 1
    assert np.isclose(abs(k.agent_heading), np.pi, atol=1e-6)
    assert np.allclose(k.agent_position, [-1.0, 1.0, 1.2], atol=1e-9)


def test_elevation_is_degrees():
    # Upstream converts elevation via np.radians -> 90 means straight up.
    k = _kin_free()
    k.move(0.0, 90.0, 1.0)
    assert np.allclose(k.agent_position, [0.0, 0.0, 2.2], atol=1e-9)


# ==================== occupancy / freemap ====================
#
# Synthetic freemap convention (matches <scene>/freemap.npy):
#   row 0   = world-x coordinate of each column
#   col 0   = world-y coordinate of each row
#   interior: 0 obstacle, 1 reachable, 2 out-of-bounds
# Coordinates use a 0.125 m grid step (exactly representable in binary) so
# absolute world<->grid assertions below are exact, not approximate.

X0, Y0, STEP, N = 10.0, 20.0, 0.125, 12


def _make_freemap(interior: float = 1.0) -> np.ndarray:
    occ = np.full((N, N), interior, dtype=np.float64)
    occ[0, :] = X0 + STEP * np.arange(N)  # x axis along columns
    occ[:, 0] = Y0 + STEP * np.arange(N)  # y axis along rows
    return occ


def _kin_occ(tmp_path, occ: np.ndarray) -> FreemapKinematics:
    path = tmp_path / "freemap.npy"
    np.save(path, occ)
    k = FreemapKinematics(use_occupancy_collision=True)
    k.load_freemap(str(path))
    return k


def test_world_grid_mapping_hand_pinned(tmp_path):
    k = _kin_occ(tmp_path, _make_freemap())
    x, y = X0 + 5 * STEP, Y0 + 3 * STEP  # 10.625, 20.375 — exact doubles
    assert k.find_nearest_reachable(x, y) == (x, y, 0)


def test_free_path_no_collision_snaps_to_cell(tmp_path):
    k = _kin_occ(tmp_path, _make_freemap())
    k.set_pose(np.array([X0 + 2 * STEP + 0.01, Y0 + 2 * STEP + 0.01, 1.2]), 0.0)
    k.agent_rotation_quat = euler_angles_to_quat(np.array([0.0, 0.0, 0.0]))
    collision, dev = k.move(0.0, 0.0, 3 * STEP)  # intended lands ~cell (2, 5)
    assert collision is False
    assert dev <= np.hypot(STEP / 2, STEP / 2) + 1e-9  # snapped within half-cell
    assert k.agent_position[0] in set((X0 + STEP * np.arange(N)).tolist())
    assert k.agent_position[1] in set((Y0 + STEP * np.arange(N)).tolist())


def test_wall_forces_snap_and_collision(tmp_path):
    occ = _make_freemap()
    occ[1:, 7:10] = 0.0  # 3-column wall at x = 10.875, 11.0, 11.125
    k = _kin_occ(tmp_path, occ)
    wall_x, mid_y = X0 + 8 * STEP, Y0 + 5 * STEP
    res = k.find_nearest_reachable(wall_x, mid_y)
    assert res is not None
    rx, ry, grid_d = res
    assert grid_d > 0
    assert np.isclose(abs(rx - wall_x), 2 * STEP, atol=1e-9)  # nearest free column
    k.set_pose(np.array([wall_x - 4 * STEP, mid_y, 1.2]), 0.0)
    k.agent_rotation_quat = euler_angles_to_quat(np.array([0.0, 0.0, 0.0]))
    collision, dev = k.move(0.0, 0.0, 4 * STEP)
    assert collision is True
    assert np.isclose(dev, 2 * STEP, atol=1e-9)
    assert k.collision_occurred is True


def test_no_reachable_cell_stays_put(tmp_path):
    occ = _make_freemap(interior=0.0)  # everything blocked
    k = _kin_occ(tmp_path, occ)
    start = np.array([X0 + 5 * STEP, Y0 + 5 * STEP, 1.2])
    k.set_pose(start, 0.0)
    k.agent_rotation_quat = euler_angles_to_quat(np.array([0.0, 0.0, 0.0]))
    collision, dev = k.move(0.5, 0.0, 1.0)
    assert collision is True and dev == 0.0
    assert np.allclose(k.agent_position, start)  # position unchanged...
    assert np.isclose(k.agent_heading, 0.5, atol=1e-9)  # ...but heading rotated


def test_missing_freemap_raises():
    k = FreemapKinematics(use_occupancy_collision=True)
    with pytest.raises(RuntimeError, match="freemap"):
        k.move(0.0, 0.0, 1.0)


# ==================== world-level action mapping (no Isaac) ====================


def test_world_discrete_actions_without_backend():
    """IsaacWorld.move() needs no simulator: the kinematics are plain state."""
    from embodiedscore_envs.benchmarks.env.sim import IsaacBody, IsaacWorld
    w = IsaacWorld(IsaacBody(collision="none"))
    w.place([1.0, 2.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    pos, rot = w.pose()
    assert np.allclose(pos, [1.0, 2.0, 1.2]) and np.allclose(rot, [1.0, 0.0, 0.0, 0.0])
    assert w.move(2) is False                                   # LEFT
    assert np.isclose(w.heading(), np.radians(15.0))
    assert w.move(3) is False and np.isclose(w.heading(), 0.0)  # RIGHT undoes it
    assert w.move(1) is False                                   # FORWARD 0.25 m along +x
    assert np.allclose(w.pose()[0], [1.25, 2.0, 1.2])
    assert np.isclose(w.distance([1.25, 2.0, 1.2], [[1.25, 5.0, 0.0]]), 3.0)
    with pytest.raises(ValueError):
        w.move(0)
    w.close()
