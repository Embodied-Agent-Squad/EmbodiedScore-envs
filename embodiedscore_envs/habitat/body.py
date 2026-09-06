"""Embodiment parameters: what the agent's body and cameras are.

``Body.std()`` is the frozen standard used by the R2R-CE / RxR-CE boards
(configs/habitat/std_task.yaml in the AgentCanvas workspace): 0.25 m steps,
15 degree turns, a 1.5 m / 0.1 m agent, RGB 512x512 and depth 256x256 at
hfov 90 mounted 1.25 m up, wall sliding on. Everything in SimWorld reads its
numbers from here; nothing else in the package hard-codes them.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CameraSpec:
    width: int
    height: int
    hfov_deg: float = 90.0
    position: tuple[float, float, float] = (0.0, 1.25, 0.0)  # metres, agent frame (y up)


@dataclass(frozen=True)
class Body:
    forward_step_m: float = 0.25
    turn_deg: float = 15.0
    agent_height_m: float = 1.5
    agent_radius_m: float = 0.1
    allow_sliding: bool = True
    rgb: CameraSpec = field(default_factory=lambda: CameraSpec(512, 512))
    depth: CameraSpec = field(default_factory=lambda: CameraSpec(256, 256))

    @classmethod
    def std(cls) -> "Body":
        """The board standard (see module docstring)."""
        return cls()
