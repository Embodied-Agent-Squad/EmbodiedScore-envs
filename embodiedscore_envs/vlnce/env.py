"""VLNCEEnv — the environment body (layer L1), a plain ``gymnasium.Env``.

What it does: hold the episode list and the body, put the agent at an
episode's start, execute one discrete action per ``step``, and report facts
(observation + position / rotation / heading / collided / distance_to_goal).

What it deliberately does not do: count steps or enforce a budget
(``TimeLimit``), compute Success / SPL / nDTW (``VLNCEMetrics``), normalise
depth (``NormalizeDepth``), keep a "next episode" cursor (episodes are chosen
through ``reset(options=...)`` or sampled with the env's RNG), or put the
instruction into the observation (it is episode-constant and travels in the
reset ``info``).

Actions: Discrete(4) — 0 STOP, 1 FORWARD, 2 LEFT, 3 RIGHT. STOP does not move
and sets ``terminated``. ``reward`` is always 0.0; this is an evaluation
environment, reward shaping belongs in a wrapper.
"""

from __future__ import annotations

from typing import Any, Iterable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..habitat.body import Body
from ..habitat.world import FORWARD, LEFT, RIGHT, SimWorld
from .dataset import Episode, load_episodes, scene_paths


class Act:
    STOP = 0
    FORWARD = FORWARD
    LEFT = LEFT
    RIGHT = RIGHT


class VLNCEEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(
        self,
        dataset: str = "r2r",
        split: str = "val_unseen",
        data_root: str | None = None,
        scene_root: str | None = None,
        body: Body | None = None,
        languages: Iterable[str] | None = ("en-US", "en-IN"),
        gpu_id: int = 0,
        render_mode: str | None = None,
    ) -> None:
        self.dataset = dataset
        self.split = split
        self.body = body or Body.std()
        self.render_mode = render_mode
        self._scene_root = scene_root
        self.episodes: tuple[Episode, ...] = tuple(load_episodes(dataset, split, data_root, languages))
        if not self.episodes:
            raise ValueError(f"no episodes for {dataset}/{split} (languages={languages})")
        self._by_id = {e.episode_id: e for e in self.episodes}
        self._world = SimWorld(self.body, gpu_id=gpu_id)   # simulator is created on first reset
        self._episode: Episode | None = None
        self._stop_called = False
        self._last_rgb: np.ndarray | None = None

        rgb, depth = self.body.rgb, self.body.depth
        self.observation_space = spaces.Dict({
            "rgb": spaces.Box(0, 255, shape=(rgb.height, rgb.width, 3), dtype=np.uint8),
            "depth": spaces.Box(0.0, np.inf, shape=(depth.height, depth.width, 1), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(4)

    # ---- episode selection -------------------------------------------------
    @property
    def episode(self) -> Episode:
        if self._episode is None:
            raise RuntimeError("VLNCEEnv: call reset() first")
        return self._episode

    def _select(self, options: dict[str, Any] | None) -> Episode:
        options = options or {}
        if "episode" in options:
            return self.episodes[int(options["episode"])]
        if "episode_id" in options:
            return self._by_id[str(options["episode_id"])]
        return self.episodes[int(self.np_random.integers(len(self.episodes)))]

    # ---- gym API -------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        ep = self._select(options)
        glb, navmesh = scene_paths(ep, self._scene_root)
        self._world.load_scene(glb, navmesh)
        self._world.place(ep.start_position, ep.start_rotation)
        self._episode = ep
        self._stop_called = False
        obs = self._observe()
        info = self._facts(collided=False)
        info["episode"] = ep.as_dict()
        return obs, info

    def step(self, action: int):
        ep = self.episode
        if self._stop_called:
            raise RuntimeError("VLNCEEnv: episode is over (STOP was called); reset() first")
        a = int(action)
        if a == Act.STOP:
            self._stop_called = True
            collided = False
        elif a in (Act.FORWARD, Act.LEFT, Act.RIGHT):
            collided = self._world.move(a)
        else:
            raise ValueError(f"action must be in 0..3, got {action!r}")
        obs = self._observe()
        info = self._facts(collided=collided)
        del ep
        return obs, 0.0, self._stop_called, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return None if self._last_rgb is None else self._last_rgb.copy()
        return None

    def close(self) -> None:
        self._world.close()

    # ---- helpers -------------------------------------------------------------
    def _observe(self) -> dict[str, np.ndarray]:
        obs = self._world.observe()
        self._last_rgb = obs["rgb"]
        return obs

    def _facts(self, *, collided: bool) -> dict[str, Any]:
        pos, rot = self._world.pose()
        return {
            "position": pos.tolist(),
            "rotation": rot.tolist(),
            "heading": self._world.heading(),
            "collided": bool(collided),
            "distance_to_goal": self._world.geodesic(pos, self.episode.goal_position),
            "stop_called": self._stop_called,
        }

    @property
    def world(self) -> SimWorld:
        """The engine facade — for wrappers and tools that need geodesics or the pathfinder."""
        return self._world
