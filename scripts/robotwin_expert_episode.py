"""Drive a RoboTwin episode with RoboTwin's own scripted expert, THROUGH the
package's macro pose protocol — the proof that ``RobotwinPoseEnv`` can reach
``success = 1.0``.

RoboTwin ships one expert per task: ``Base_Task.play_once`` reads actor poses
straight out of the simulator (privileged), computes end-effector waypoints
with ``grasp_actor`` / ``place_actor`` / ``move_by_displacement``, and hands
them to ``Base_Task.move``, which executes them by servoing joints directly
(``take_dense_action``). That is NOT the protocol an agent gets.

This script keeps the expert's WAYPOINTS and throws away its execution:
``Base_Task.move`` is replaced with a shim that turns each ``Action`` into one
``RobotwinPoseEnv.step`` — an absolute end-effector target per arm, or a
gripper hold — so the whole episode runs on the same interface the bareES arm
surface drives. Because the shim really moves the arm, every waypoint the
expert computes afterwards sees the state the macro protocol actually
produced.

    python scripts/robotwin_expert_episode.py --line robotwin-clean --split mini --episode 0

Needs the RoboTwin checkout (``EMBODIEDSCORE_ROBOTWIN_ROOT``) and its
dependencies; see INSTALL-robotwin.md. Exit code 0 only when the episode's
metrics report ``success == 1.0``.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def robotwin_gripper_to_action(value: float) -> float:
    """RoboTwin's normalised gripper value (0 closed .. 1 open) to the macro
    protocol's field (+1 closed .. -1 open) — the inverse of
    ``robotwin_env.gripper_to_robotwin``."""
    return float(1.0 - 2.0 * float(value))


def _rotvec(quat_wxyz) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    return Rotation.from_quat([x, y, z, w]).as_rotvec()


def _split_by_arm(actions_by_arm1, actions_by_arm2):
    """``Base_Task.move``'s own pairing (``envs/_base_task.py:901``): each
    argument is ``(arm_tag, [Action, ...])``; the two lists are zipped so one
    tick can command both arms."""
    lists = {"left": [], "right": []}
    for pair in (actions_by_arm1, actions_by_arm2):
        if not pair:
            continue
        tag, actions = pair
        if tag is None or not actions:
            continue
        lists[str(tag)] = list(actions)
    width = max(len(lists["left"]), len(lists["right"]))
    for arm in ("left", "right"):
        lists[arm] += [None] * (width - len(lists[arm]))
    return [(lists["left"][i], lists["right"][i]) for i in range(width)]


class ExpertThroughPoseEnv:
    """Replaces ``Base_Task.move`` so the expert's waypoints run through the
    package's macro protocol instead of RoboTwin's joint servo."""

    def __init__(self, env, verbose: bool = True) -> None:
        from embodiedscore_envs.benchmarks.env.robotwin_env import ARM_ACTION

        self.env = env
        self.base = env.unwrapped
        self.world = self.base.world
        self.arm_action = ARM_ACTION
        self.verbose = verbose
        self.steps = 0
        self.over = False
        self.log: list[str] = []

    def _block(self, arm: str, action) -> list[float]:
        """One arm's 7 numbers: a target pose, or a hold (``inf``)."""
        hold = [np.inf, np.inf, np.inf, 0.0, 0.0, 0.0]
        keep = robotwin_gripper_to_action(self.world.gripper_value(arm))
        if action is None:
            return hold + [keep]
        if action.action == "gripper":
            return hold + [robotwin_gripper_to_action(action.target_gripper_pos)]
        pose = np.asarray(action.target_pose, dtype=np.float64).reshape(7)
        return list(pose[:3]) + list(_rotvec(pose[3:])) + [keep]

    def move(self, actions_by_arm1, actions_by_arm2=None, save_freq=-1) -> bool:
        if self.over:
            return False
        for left, right in _split_by_arm(actions_by_arm1, actions_by_arm2):
            vector = np.asarray(self._block("left", left) + self._block("right", right), dtype=np.float64)
            obs, reward, terminated, truncated, info = self.env.step(vector)
            self.steps += 1
            metrics = info.get("metrics") or {}
            line = (f"  step {self.steps:>3}  L {'move' if left is not None and left.action == 'move' else '-':<5}"
                    f" R {'move' if right is not None and right.action == 'move' else '-':<5}"
                    f" converged {str(info['converged']):<5} err {info['position_error_m']:.4f} m"
                    f"  ticks {info['ticks']:>3}  success {metrics.get('success', 0.0)}")
            self.log.append(line)
            if self.verbose:
                print(line, flush=True)
            if terminated or truncated:
                self.over = True
                return False
        return True


def run(line: str, split: str, episode: int, verbose: bool = True) -> dict:
    os.environ.setdefault("SAPIEN_HEADLESS", "1")
    import torch.multiprocessing as mp

    try:                                    # RoboTwin's planner runs in child processes
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    import embodiedscore_envs as es

    env = es.make(line, split)
    try:
        base = env.unwrapped
        ep = base.episodes[episode]
        if verbose:
            print(f"line {line} / {split} episode {episode}: {ep.episode_id}")
            print(f"instruction: {ep.instruction}")
        obs, info = env.reset(options={"episode": episode, "need_plan": True})
        if verbose:
            print(f"seed {info['seed']} (after {info['seed_attempts']} attempt(s)), "
                  f"tick cap {info['step_budget']}, macro budget {es.benchmark(line).max_episode_steps}")

        task = base.world.task
        driver = ExpertThroughPoseEnv(env, verbose=verbose)
        task.move = driver.move                          # the expert's waypoints, our protocol
        base.world.play_expert()

        # one final no-op hold so the metric wrapper reports the latched result
        _, _, _, _, last = env.step(
            np.asarray(driver._block("left", None) + driver._block("right", None), dtype=np.float64))
        metrics = dict(last.get("metrics") or {})
        metrics["macro_steps"] = driver.steps + 1
        metrics["seed"] = info["seed"]
        metrics["episode_id"] = ep.episode_id
        if verbose:
            print(f"\nRESULT {ep.episode_id}: {metrics}")
        return metrics
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--line", default="robotwin-clean")
    parser.add_argument("--split", default="mini")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    metrics = run(args.line, args.split, args.episode, verbose=not args.quiet)
    ok = float(metrics.get("success", 0.0)) == 1.0
    print("SUCCESS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
