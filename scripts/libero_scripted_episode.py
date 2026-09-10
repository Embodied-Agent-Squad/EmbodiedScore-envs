#!/usr/bin/env python
"""A scripted pick-and-place on any LIBERO-engine line, driven by privileged
information — the proof that a line's macro protocol can reach ``success=1.0``.

It is not a policy and not a baseline: it reads the BDDL goal predicate and
the simulator's own geometry (object poses from ``info["objects"]``, mesh
bounds from the MuJoCo model), then issues nine macro steps of the standard
protocol (``LiberoPoseEnv``: absolute end-effector targets, ``hold_action``
for the gripper). Nothing an agent may see is used to choose the targets.

The primitive is LIBERO's rim grasp: the akita-bowl family is wider than the
Panda gripper can span, so the gripper straddles the RIM — offset from the
object centre by half its extent along the finger axis (world y at the home
orientation), lowered to a third of the way down from the rim. The object
therefore hangs off-centre in the grip, so the carry offset (where the object
sits relative to the end-effector) is MEASURED after the lift and subtracted
from the destination: the release puts the object's centre over the
destination's centre and its base just above the destination's top.

    python scripts/libero_scripted_episode.py libero-pro-spatial --split mini --episode 0
    python scripts/libero_scripted_episode.py libero-plus-camera --split mini --episode 0

Exit status 0 on ``success == 1.0``. Needs the line's ``libero`` package and
EGL (``MUJOCO_GL=egl``, set here when unset).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import numpy as np

APPROACH_M = 0.08          # height above the grasp pose the gripper descends from
LIFT_M = 0.15              # how far the object is lifted before it travels
CLEARANCE_M = 0.01         # gap between the carried object's base and the destination's top at release
RIM_FRACTION = 0.50        # grasp offset from the centre, as a fraction of the object's extent
DEPTH_FRACTION = 0.35      # grasp height below the object's top, as a fraction of its height
GRASP_AXES = ("-y", "+y")  # sides of the rim to try, in order
PLACE_PREDICATES = ("on", "in")


def _rotvec(quat_wxyz: list[float]) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    w, x, y, z = quat_wxyz
    return Rotation.from_quat([x, y, z, w]).as_rotvec()


def mesh_bounds(env: Any, body_prefix: str) -> tuple[np.ndarray, np.ndarray]:
    """World-frame axis-aligned bounds of every geom of a BDDL object, from
    the mesh vertices (``geom_size`` is a bounding sphere for meshes and
    overstates a bowl by half its width)."""
    sim = env.unwrapped.world.sim
    model, data = sim.model, sim.data
    raw = getattr(model, "_model", model)
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for gi in range(model.ngeom):
        name = model.body_id2name(model.geom_bodyid[gi])
        if name is None or not name.startswith(body_prefix):
            continue
        pos = np.asarray(data.geom_xpos[gi], dtype=np.float64)
        mat = np.asarray(data.geom_xmat[gi], dtype=np.float64).reshape(3, 3)
        if model.geom_type[gi] == 7:      # mjGEOM_MESH
            mid = model.geom_dataid[gi]
            start, count = raw.mesh_vertadr[mid], raw.mesh_vertnum[mid]
            verts = np.asarray(raw.mesh_vert[start:start + count], dtype=np.float64).reshape(-1, 3)
        else:
            s = np.asarray(model.geom_size[gi], dtype=np.float64)
            verts = np.array([[a, b, c] for a in (-s[0], s[0]) for b in (-s[1], s[1]) for c in (-s[2], s[2])])
        world = verts @ mat.T + pos
        lo = np.minimum(lo, world.min(axis=0))
        hi = np.maximum(hi, world.max(axis=0))
    if not np.isfinite(lo).all():
        raise RuntimeError(f"no geoms for object {body_prefix!r}")
    return lo, hi


def place_predicate(predicates: list[list[str]]) -> tuple[str, str]:
    for pred in predicates:
        if len(pred) >= 3 and str(pred[0]).lower() in PLACE_PREDICATES:
            return str(pred[1]), str(pred[2])
    raise SystemExit(f"no {'/'.join(PLACE_PREDICATES)} predicate to script in {predicates}")


class Driver:
    """The macro steps, and the facts each one returned."""

    def __init__(self, env: Any, rotation: np.ndarray, verbose: bool) -> None:
        self.env, self.rotation, self.verbose = env, rotation, verbose
        self.info: dict[str, Any] = {}

    def move(self, xyz: np.ndarray, gripper: float) -> dict[str, Any]:
        action = np.concatenate([np.asarray(xyz, dtype=np.float64), self.rotation, [gripper]])
        _, _, _, _, self.info = self.env.step(action)
        self._log(f"move  -> {np.round(xyz, 3)} grip={gripper:+.0f} "
                  f"converged={self.info['converged']} err={self.info['position_error_m']:.4f}")
        return self.info

    def hold(self, gripper: float) -> dict[str, Any]:
        from embodiedscore_envs.benchmarks.env import LiberoPoseEnv

        _, _, _, _, self.info = self.env.step(LiberoPoseEnv.hold_action(gripper))
        self._log(f"hold  grip={gripper:+.0f} opening={self.info['gripper_open']:.1f}")
        return self.info

    def _log(self, line: str) -> None:
        if self.verbose:
            print(f"  {line}  ticks={self.info['ticks']} success={self.info['metrics']['success']}")


def grasp_pose(lo: np.ndarray, hi: np.ndarray, axis: str) -> np.ndarray:
    centre = (lo + hi) / 2.0
    extent = hi - lo
    sign = -1.0 if axis.startswith("-") else 1.0
    index = {"x": 0, "y": 1}[axis[-1]]
    pose = np.array([centre[0], centre[1], hi[2] - extent[2] * DEPTH_FRACTION])
    pose[index] += sign * extent[index] * RIM_FRACTION
    return pose


def run_episode(line: str, split: str, episode: int, verbose: bool = True) -> tuple[float, dict[str, Any]]:
    import embodiedscore_envs as es

    env = es.make(line, split)
    try:
        for axis in GRASP_AXES:
            obs, info = env.reset(options={"episode": episode})
            target_name, dest_name = place_predicate(info["goal"]["predicates"])
            drive = Driver(env, _rotvec(info["eef_rotation"]), verbose)
            if verbose:
                print(f"{line} / {split} / episode {episode}: {info['episode']['episode_id']}")
                print(f"  instruction {info['episode']['instruction']!r}")
                print(f"  goal        put {target_name} on {dest_name}; grasping the {axis} rim")
            grasp = grasp_pose(*mesh_bounds(env, target_name), axis)
            drive.move(grasp + [0, 0, APPROACH_M], -1)
            drive.move(grasp, -1)
            drive.hold(+1)
            lifted = drive.move(np.asarray(drive.info["eef_position"]) + [0, 0, LIFT_M], +1)
            eef = np.asarray(lifted["eef_position"], dtype=np.float64)
            held = np.asarray(lifted["objects"][target_name]["position"], dtype=np.float64)
            if held[2] < grasp[2]:         # the object stayed on the table: the other side of the rim, then
                if verbose:
                    print(f"  the {axis} rim did not hold (object base {held[2]:.3f}); trying the other side")
                continue
            carry = held - eef             # where the object hangs relative to the end-effector
            dest_lo, dest_hi = mesh_bounds(env, dest_name)
            dest_centre = (dest_lo + dest_hi) / 2.0
            release = np.array([dest_centre[0] - carry[0], dest_centre[1] - carry[1],
                                dest_hi[2] + CLEARANCE_M - carry[2]])
            drive.move(np.array([release[0], release[1], eef[2]]), +1)
            drive.move(release, +1)
            drive.hold(-1)
            drive.move(np.array([release[0], release[1], eef[2]]), -1)
            info = drive.hold(-1)          # let the released object settle; success latches on any tick
            return float(info["metrics"]["success"]), dict(info["metrics"])
        raise SystemExit("no side of the rim held the object")
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("line", help="a libero-engine line, standard variant (macro protocol)")
    parser.add_argument("--split", default="mini")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    success, metrics = run_episode(args.line, args.split, args.episode, verbose=not args.quiet)
    print(f"{'SUCCESS' if success == 1.0 else 'FAILURE'}  {args.line}/{args.split}/{args.episode}  {metrics}")
    return 0 if success == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
