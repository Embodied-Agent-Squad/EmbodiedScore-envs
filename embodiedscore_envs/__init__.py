"""EmbodiedScore environments — every habitat benchmark of the workspace on one
simulator (habitat-sim 0.3.3, EmbodiedScore-habitat) behind Gymnasium 1.3.

    import embodiedscore_envs as es
    env = es.make("vlnce-r2r", "val_unseen")          # the standard stack
    obs, info = env.reset(options={"episode": 0})
    obs, r, terminated, truncated, info = env.step(es.Act.FORWARD)
    info["metrics"]

Gymnasium ids (``gym.make(id, split=...)`` gives the bare body) are registered
on import, one per benchmark declaration in ``embodiedscore_envs.benchmarks``.
"""

from gymnasium.envs.registration import register

from .benchmarks import BENCHMARKS, benchmark, build_env, make
from .benchmarks.env import Act

__version__ = "0.2.0"

for _b in BENCHMARKS.values():
    register(id=_b.gym_id, entry_point="embodiedscore_envs.benchmarks:build_env",
             max_episode_steps=_b.max_episode_steps, kwargs={"benchmark_name": _b.name})
del _b

__all__ = ["BENCHMARKS", "benchmark", "build_env", "make", "Act", "__version__"]
