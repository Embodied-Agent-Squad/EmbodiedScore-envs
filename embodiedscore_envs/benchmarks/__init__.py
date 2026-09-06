"""Layer L2 — the benchmark declarations, and ``make()`` that turns one into a
Gymnasium stack. Each member module exposes a ``BENCHMARKS`` tuple; nothing
else in the package knows their names."""

from __future__ import annotations

import importlib
from typing import Any

import gymnasium as gym

from .env import Benchmark, DepthClip, DynamicTimeLimit, HabitatEnv, HabitatPoseEnv

_MEMBERS = ("vlnce", "ivlnce", "objectnav", "goat", "hmeqa", "express")

BENCHMARKS: dict[str, Benchmark] = {}
for _m in _MEMBERS:
    try:
        _mod = importlib.import_module(f"{__name__}.{_m}")
    except ModuleNotFoundError as _e:   # a member not written yet
        if _e.name != f"{__name__}.{_m}":
            raise
        continue
    for _b in _mod.BENCHMARKS:
        if _b.name in BENCHMARKS:
            raise RuntimeError(f"duplicate benchmark name {_b.name}")
        BENCHMARKS[_b.name] = _b
del _m, _mod, _b


def benchmark(name: str) -> Benchmark:
    try:
        return BENCHMARKS[name]
    except KeyError:
        raise KeyError(f"unknown benchmark {name!r}; known: {sorted(BENCHMARKS)}") from None


def build_env(benchmark_name: str, split: str, data_root: str | None = None, scene_root: str | None = None,
              body: Any = None, actions: Any = None, gpu_id: int = 0, sim_seed: int = 42,
              render_mode: str | None = None, **loader_kwargs: Any) -> gym.Env:
    """The bare body for a benchmark (what the gym ids point at). ``body`` /
    ``actions`` override the declaration (a different rig or action table on
    the same episodes and measures — how the legacy nodesets' per-graph
    sensor knobs and upstream RxR's body are served)."""
    b = benchmark(benchmark_name)
    episodes = b.episodes(split, data_root=data_root, scene_root=scene_root, **loader_kwargs)
    if not episodes:
        raise ValueError(f"{benchmark_name}/{split}: no episodes")
    common = dict(episodes=episodes, body=body or b.body, budget=b.budget, dtg_policy=b.dtg_policy, gpu_id=gpu_id,
                  sim_seed=sim_seed, render_mode=render_mode)
    if b.pose:
        return HabitatPoseEnv(snap=b.pose_snap, **common)
    return HabitatEnv(actions=actions or b.actions, **common)


def make(name: str, split: str, *, metrics: bool = True, depth: bool = True, depth_spec: Any = None,
         metric_overrides: dict[str, Any] | None = None, **kwargs: Any) -> gym.Env:
    """The standard stack: gym.make (PassiveEnvChecker / OrderEnforcing / TimeLimit
    when the benchmark has a static budget) -> the benchmark's metric wrapper ->
    DepthClip -> DynamicTimeLimit (per-episode budgets). ``kwargs`` go to
    :func:`build_env` (data_root, scene_root, body, actions, gpu_id, loader
    options); ``depth_spec`` replaces the declared depth post-processing;
    ``metric_overrides`` (e.g. ``success_distance``) reach the metric factory."""
    b = benchmark(name)
    env = gym.make(b.gym_id, split=split, **kwargs)
    if metrics and b.metrics is not None:
        env = b.metrics(env, **(metric_overrides or {}))
    spec = depth_spec if depth_spec is not None else b.depth
    body = kwargs.get("body") or b.body
    if depth and spec is not None and body.depth is not None:
        env = DepthClip(env, spec.min_m, spec.max_m, spec.normalize)
    if b.budget is not None and b.truncate_at_budget:
        env = DynamicTimeLimit(env)
    return env


__all__ = ["BENCHMARKS", "benchmark", "build_env", "make"]
