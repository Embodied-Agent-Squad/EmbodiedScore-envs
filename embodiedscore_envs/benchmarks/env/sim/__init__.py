"""Layer L0 — the simulator facades, one per engine.

``habitat/``  habitat-sim 0.3.3 in process: Body · SceneRef · SimWorld (+ Follower).
``isaac/``    Isaac Sim 5.1 out of process: IsaacBody · IsaacSceneRef · IsaacWorld
              (render worker under Isaac's python, freemap kinematics on this side).

The engines never import each other. The pure-data types of both are
importable without either simulator installed; ``SimWorld`` and its
companions (the habitat_sim importers) resolve lazily, so ``import
embodiedscore_envs`` works on a host that has only Isaac (or neither) — a
habitat line then fails at construction with the ImportError that names the
build to install.
"""

from __future__ import annotations

from .habitat import Body, CameraSpec, NavMesh, SceneRef
from .isaac import IsaacBody, IsaacCameraSpec, IsaacSceneRef, IsaacSettings, IsaacWorld

_HABITAT_LAZY = ("SimWorld", "Follower", "FollowerError", "FORWARD", "LEFT", "RIGHT", "LOOK_UP", "LOOK_DOWN")


def __getattr__(name: str):
    if name in _HABITAT_LAZY:
        from .habitat import world
        return getattr(world, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Body", "CameraSpec", "NavMesh", "SceneRef", "SimWorld", "Follower", "FollowerError",
           "FORWARD", "LEFT", "RIGHT", "LOOK_UP", "LOOK_DOWN",
           "IsaacBody", "IsaacCameraSpec", "IsaacSceneRef", "IsaacSettings", "IsaacWorld"]
