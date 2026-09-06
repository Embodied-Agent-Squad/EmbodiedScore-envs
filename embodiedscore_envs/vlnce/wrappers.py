"""Consumer-side wrappers and the standard stack assembler."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .metrics import VLNCEMetrics


class NormalizeDepth(gym.ObservationWrapper):
    """clip(depth, lo, hi) / hi, mirroring habitat-lab 0.1.7's
    HabitatSimDepthSensor with NORMALIZE_DEPTH=True (MIN_DEPTH 0, MAX_DEPTH 10)."""

    def __init__(self, env: gym.Env, min_depth: float = 0.0, max_depth: float = 10.0) -> None:
        super().__init__(env)
        self.lo, self.hi = float(min_depth), float(max_depth)
        d = env.observation_space["depth"]
        new = dict(env.observation_space.spaces)
        new["depth"] = spaces.Box(0.0, 1.0, shape=d.shape, dtype=np.float32)
        self.observation_space = spaces.Dict(new)

    def observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        d = np.clip(obs["depth"], self.lo, self.hi)
        obs = dict(obs)
        obs["depth"] = ((d - self.lo) / (self.hi - self.lo)).astype(np.float32)
        return obs


def make_vlnce(dataset: str = "r2r", split: str = "val_unseen", *, max_episode_steps: int = 500,
               normalize_depth: bool = True, metrics: bool = True, **env_kwargs: Any) -> gym.Env:
    """The board stack: PassiveEnvChecker / OrderEnforcing / TimeLimit (via gym.make)
    -> VLNCEMetrics -> NormalizeDepth (outermost, observation only)."""
    env_id = {"r2r": "EmbodiedScore/VLNCE-R2R-v0", "rxr": "EmbodiedScore/VLNCE-RxR-v0"}[dataset]
    env = gym.make(env_id, split=split, max_episode_steps=max_episode_steps, **env_kwargs)
    if metrics:
        env = VLNCEMetrics(env, data_root=env_kwargs.get("data_root"))
    if normalize_depth:
        env = NormalizeDepth(env)
    return env
