"""VLNCEMetrics — the seven board metrics as a gymnasium Wrapper (layer L2).

Formulas are transcribed from habitat-lab 0.1.7 (``habitat/tasks/nav/nav.py``:
DistanceToGoal, Success, SPL) and VLN-CE (``habitat_extensions/measures.py``:
NDTW, PathLength, OracleSuccess, StepsTaken), including their quirks:

* ``distance_to_goal``  geodesic(agent, goal), evaluated at reset and after every step
* ``success``           stop_called and distance_to_goal < success_distance
* ``spl``               success * d0 / max(d0, L); d0 = distance_to_goal at reset
                        (from the simulator, not the dataset), L = sum of
                        *euclidean* step lengths
* ``ndtw``              fastdtw(agent locations, reference locations, euclidean);
                        exp(-dtw / (len(reference) * success_distance)); the
                        agent location list starts with the start position and
                        skips a step whose position equals the previous one
* ``path_length``       the same euclidean L (its docstring upstream says geodesic;
                        the code is euclidean, and the code is what scored the boards)
* ``oracle_success``    1.0 once distance_to_goal < success_distance at any point,
                        reset included
* ``steps_taken``       number of step() calls, STOP included

The wrapper reads only the facts the body reports (``position``,
``distance_to_goal``, ``stop_called``) and writes ``info["metrics"]`` after
reset and after every step. When d0 is not finite (goal unreachable on the
navmesh — this happens for one RxR rand100 episode) ``info["metrics_valid"]``
is False and spl/ndtw are whatever the formulas give.
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from fastdtw import fastdtw

from .dataset import load_gt_locations

METRIC_KEYS = ("distance_to_goal", "success", "spl", "ndtw", "path_length", "oracle_success", "steps_taken")


def _euclid(a, b) -> float:
    return float(np.linalg.norm(np.array(b) - np.array(a), ord=2))


class VLNCEMetrics(gym.Wrapper):
    def __init__(self, env: gym.Env, gt_locations: dict[str, list[list[float]]] | None = None,
                 data_root: str | None = None, success_distance: float = 3.0) -> None:
        super().__init__(env)
        base = env.unwrapped
        self._gt = gt_locations if gt_locations is not None else load_gt_locations(base.dataset, base.split, data_root)
        self._sd = float(success_distance)
        self._m: dict[str, float] = {}
        self._d0 = math.inf
        self._prev: list[float] = []
        self._L = 0.0
        self._locations: list[list[float]] = []
        self._gt_locs: list[list[float]] = []

    @property
    def metrics(self) -> dict[str, float]:
        return dict(self._m)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        ep_id = info["episode"]["episode_id"]
        pos = list(info["position"])
        self._d0 = float(info["distance_to_goal"])
        self._prev = pos
        self._L = 0.0
        self._locations = []
        self._gt_locs = self._gt[ep_id]
        self._m = {
            "distance_to_goal": self._d0,
            "success": 0.0,
            "spl": 0.0,
            "ndtw": self._ndtw_update(pos),
            "path_length": 0.0,
            "oracle_success": float(self._d0 < self._sd),
            "steps_taken": 0.0,
        }
        return obs, self._annotate(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        pos = list(info["position"])
        dtg = float(info["distance_to_goal"])
        stop = bool(info["stop_called"])
        self._L += _euclid(self._prev, pos)
        self._prev = pos
        success = float(stop and dtg < self._sd)
        self._m = {
            "distance_to_goal": dtg,
            "success": success,
            "spl": success * (self._d0 / max(self._d0, self._L)) if math.isfinite(self._d0) else 0.0,
            "ndtw": self._ndtw_update(pos),
            "path_length": self._L,
            "oracle_success": float(self._m["oracle_success"] or dtg < self._sd),
            "steps_taken": self._m["steps_taken"] + 1.0,
        }
        return obs, reward, terminated, truncated, self._annotate(info)

    # ---- pieces --------------------------------------------------------------
    def _ndtw_update(self, pos: list[float]) -> float:
        if self._locations and pos == self._locations[-1]:
            return self._m.get("ndtw", 0.0)
        self._locations.append(pos)
        dtw_distance = fastdtw(self._locations, self._gt_locs, dist=_euclid)[0]
        return float(np.exp(-dtw_distance / (len(self._gt_locs) * self._sd)))

    def _annotate(self, info: dict[str, Any]) -> dict[str, Any]:
        info["metrics"] = dict(self._m)
        info["metrics_valid"] = math.isfinite(self._d0)
        return info
