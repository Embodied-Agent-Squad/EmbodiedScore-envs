"""SimWorld — the engine facade (layer L0).

Owns exactly one ``habitat_sim.Simulator`` and exposes six primitives:
load_scene / place / move / pose / geodesic / observe. It has no notion of
episode, task, success or metrics; those live above it. Every number that
defines the embodiment comes from a :class:`Body`.

Coordinate conventions are habitat-sim's: metres, y up, the agent faces -z.
Rotations cross this boundary as ``[x, y, z, w]`` quaternion coefficients
(the order VLN-CE datasets store ``start_rotation`` in).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .body import Body, CameraSpec

try:
    import habitat_sim
    from habitat_sim.utils.common import quat_from_coeffs, quat_to_coeffs
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "embodiedscore_envs needs habitat-sim 0.3.3; build it from "
        "https://github.com/Embodied-Agent-Squad/EmbodiedScore-habitat (BUILD.md)"
    ) from e

# Discrete movement actions. STOP is not a movement and never reaches SimWorld.
FORWARD, LEFT, RIGHT = 1, 2, 3
_ACTION_NAMES = {FORWARD: "move_forward", LEFT: "turn_left", RIGHT: "turn_right"}


def _camera(uuid: str, kind: Any, spec: CameraSpec) -> Any:
    s = habitat_sim.CameraSensorSpec()
    s.uuid = uuid
    s.sensor_type = kind
    s.resolution = [spec.height, spec.width]
    s.hfov = spec.hfov_deg
    s.position = list(spec.position)
    return s


class SimWorld:
    def __init__(self, body: Body | None = None, gpu_id: int = 0) -> None:
        self.body = body or Body.std()
        self.gpu_id = gpu_id
        self._sim: Any = None
        self._agent: Any = None
        self._scene: str | None = None
        self._last_obs: dict[str, np.ndarray] | None = None

    # ---- lifecycle ---------------------------------------------------------
    def _configuration(self, scene_glb: str) -> Any:
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = scene_glb
        sim_cfg.gpu_device_id = self.gpu_id
        sim_cfg.allow_sliding = self.body.allow_sliding
        sim_cfg.enable_physics = False
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.height = self.body.agent_height_m
        agent_cfg.radius = self.body.agent_radius_m
        agent_cfg.sensor_specifications = [
            _camera("rgb", habitat_sim.SensorType.COLOR, self.body.rgb),
            _camera("depth", habitat_sim.SensorType.DEPTH, self.body.depth),
        ]
        act = habitat_sim.agent.ActionSpec
        amt = habitat_sim.agent.ActuationSpec
        agent_cfg.action_space = {
            FORWARD: act("move_forward", amt(amount=self.body.forward_step_m)),
            LEFT: act("turn_left", amt(amount=self.body.turn_deg)),
            RIGHT: act("turn_right", amt(amount=self.body.turn_deg)),
        }
        return habitat_sim.Configuration(sim_cfg, [agent_cfg])

    def load_scene(self, scene_glb: str, navmesh: str | None = None) -> None:
        """Make ``scene_glb`` the active scene (no-op if it already is) and load
        its navmesh. The navmesh is taken as shipped — never recomputed — which
        is the habitat-lab 0.1.7 behaviour the VLN-CE numbers were produced with."""
        if self._sim is None:
            self._sim = habitat_sim.Simulator(self._configuration(scene_glb))
        elif scene_glb != self._scene:
            self._sim.reconfigure(self._configuration(scene_glb))
        self._scene = scene_glb
        self._agent = self._sim.initialize_agent(0)
        if navmesh is not None and not self._sim.pathfinder.load_nav_mesh(navmesh):
            raise RuntimeError(f"could not load navmesh: {navmesh}")
        self._last_obs = None

    def close(self) -> None:
        if self._sim is not None:
            self._sim.close()
            self._sim = None
            self._agent = None
            self._scene = None
            self._last_obs = None

    def __enter__(self) -> "SimWorld":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @property
    def scene(self) -> str | None:
        return self._scene

    @property
    def pathfinder(self) -> Any:
        self._require()
        return self._sim.pathfinder

    def _require(self) -> None:
        if self._sim is None:
            raise RuntimeError("SimWorld: call load_scene() first")

    # ---- primitives --------------------------------------------------------
    def place(self, position: Any, rotation_xyzw: Any) -> None:
        self._require()
        state = habitat_sim.AgentState()
        state.position = np.asarray(position, dtype=np.float32)
        state.rotation = quat_from_coeffs(np.asarray(rotation_xyzw, dtype=np.float64))
        self._agent.set_state(state)
        self._last_obs = None

    def move(self, action: int) -> bool:
        """Execute one discrete movement; return whether it collided (was cut
        short by the navmesh filter). Rendering happens here as a side effect
        of habitat's step; ``observe()`` returns that frame without re-rendering."""
        self._require()
        if action not in _ACTION_NAMES:
            raise ValueError(f"SimWorld.move: action must be 1/2/3, got {action!r}")
        obs = self._sim.step(action)
        self._last_obs = obs
        return bool(obs["collided"])

    def pose(self) -> tuple[np.ndarray, np.ndarray]:
        """(position[3] float32, rotation[x, y, z, w] float64)."""
        self._require()
        st = self._agent.get_state()
        return np.asarray(st.position, dtype=np.float32), np.asarray(quat_to_coeffs(st.rotation), dtype=np.float64)

    def heading(self) -> float:
        """Yaw in radians, habitat-lab HeadingSensor convention (0 = facing -z)."""
        self._require()
        st = self._agent.get_state()
        q = st.rotation
        fwd = np.quaternion(0, 0, 0, -1)
        v = (q * fwd * q.inverse()).imag
        return float(math.atan2(v[0], -v[2]))

    def geodesic(self, a: Any, b: Any) -> float:
        """Navmesh geodesic distance a -> b; ``inf`` when unreachable."""
        self._require()
        path = habitat_sim.ShortestPath()
        path.requested_start = np.asarray(a, dtype=np.float32)
        path.requested_end = np.asarray(b, dtype=np.float32)
        if not self._sim.pathfinder.find_path(path):
            return math.inf
        return float(path.geodesic_distance)

    def observe(self) -> dict[str, np.ndarray]:
        """{"rgb": uint8 HxWx3, "depth": float32 HxWx1 metres} at the current pose."""
        self._require()
        obs = self._last_obs if self._last_obs is not None else self._sim.get_sensor_observations()
        self._last_obs = obs
        rgb = np.ascontiguousarray(obs["rgb"][..., :3], dtype=np.uint8)
        depth = np.asarray(obs["depth"], dtype=np.float32)
        if depth.ndim == 2:
            depth = depth[..., None]
        return {"rgb": rgb, "depth": depth}
