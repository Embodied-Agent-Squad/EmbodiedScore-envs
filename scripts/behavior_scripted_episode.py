"""A scripted (privileged-information) episode on the BEHAVIOR-1K line, through
the macro protocol.

It reads what the agent never sees — the BDDL goal, the task objects' poses,
the meta link a predicate is satisfied at, the per-predicate goal status — and
drives the R1 Pro with the same three moves the agent has: ``move_base``, an
absolute end-effector target per arm, and a gripper hold. It is the proof that
the macro protocol can change the world state BEHAVIOR scores.

    # what is in this task; loads the scene, prints its facts, stops
    python scripts/behavior_scripted_episode.py --task turning_on_radio --check

    # drive it
    python scripts/behavior_scripted_episode.py --task turning_on_radio

Isaac Sim boots once per process (``omnigibson/simulator.py`` L386) and the
first load of a house takes minutes, so one run of this script is one task —
and the script spends that one load on a small search rather than a single
attempt, reporting the q_score after every move.

The plan, per goal predicate, in order:

``toggled_on``  the predicate holds when a finger link of the robot both
                touches the object and overlaps the ``togglebutton`` meta link
                for ``CAN_TOGGLE_STEPS`` (5) frames
                (``omnigibson/object_states/toggle.py``). So: drive to within
                arm's reach of that meta link, close the gripper (the fingers
                make a single small tool), and press onto the link from a few
                offsets, holding after each.
anything else   drive to the first goal object and try to pick it up: reach
                above it, descend, close, lift. Most BEHAVIOR activities need
                far more than that — the point of the run is the q_score the
                moves earn, not a solved task.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import embodiedscore_envs as es  # noqa: E402
from embodiedscore_envs.benchmarks import behavior as B  # noqa: E402
from embodiedscore_envs.benchmarks.env import BehaviorPoseEnv  # noqa: E402

REACH_M = 0.40          # how close the base is driven to a target before reaching
LIFT_M = 0.10           # how far above an object the approach pose sits
MAX_BASE_MOVES = 8


def _episode_of(env, task: str) -> int:
    for i, ep in enumerate(env.unwrapped.episodes):
        if ep.scene.task == task:
            return i
    raise SystemExit(f"{task}: not on the behavior-1k line (known: {[t for t, _ in B.TASKS]})")


def _rotvec_of(quat_wxyz) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    return Rotation.from_quat([x, y, z, w]).as_rotvec()


def _scope_names(info: dict) -> list[str]:
    """The BDDL scope names the goal predicates mention, in order."""
    placed = info.get("objects") or {}
    names: list[str] = []
    for predicate in info["goal"]["predicates"]:
        for token in predicate:
            name = str(token).lstrip("?")
            if name in placed and name not in names:
                names.append(name)
    return names


def _toggle_link(world, name: str):
    """Privileged: the world position of an object's ``togglebutton`` meta link,
    which is where ``ToggledOn`` is satisfied. None when it has no such link."""
    from omnigibson.object_states import ToggledOn

    entity = (world.task.object_scope or {}).get(name)
    if entity is None or entity.is_system or not entity.exists:
        return None
    state = entity.states.get(ToggledOn) if hasattr(entity, "states") else None
    link = getattr(state, "link", None)
    if link is None:
        return None
    return np.asarray(link.get_position_orientation()[0].cpu().numpy(), dtype=np.float64).reshape(3)


def _report(step: str, info: dict) -> None:
    m = info.get("metrics") or {}
    status = info.get("goal_status") or {}
    print(f"  {step:38s} q={m.get('q_score', 0.0):.3f} success={info['success']} "
          f"ticks={info['ticks']:6d} sat={len(status.get('satisfied', []))}"
          f"/{info.get('n_predicates', 0)} converged={info.get('converged')} "
          f"err={info.get('position_error_m')}", flush=True)


def _drive_to(env, info: dict, target: np.ndarray, reach_m: float = REACH_M) -> dict:
    """Turn toward a world point and drive until the base is within reach of it."""
    for _ in range(MAX_BASE_MOVES):
        base_pos = np.asarray(info["base_position"], dtype=np.float64)
        yaw = float(info["base_yaw_rad"])
        delta = target[:2] - base_pos[:2]
        distance = float(np.linalg.norm(delta))
        bearing = float(np.arctan2(delta[1], delta[0]))
        turn = float((bearing - yaw + np.pi) % (2 * np.pi) - np.pi)
        forward = min(max(distance - reach_m, 0.0), 1.5)
        if forward < 0.05 and abs(turn) < 0.10:
            break
        _obs, _r, term, trunc, info = env.step(BehaviorPoseEnv.base_action(forward, 0.0, turn))
        _report(f"move_base(+{forward:.2f} m, {np.degrees(turn):+.0f} deg)", info)
        if term or trunc:
            break
    return info


def _nearer_arm(info: dict, target: np.ndarray) -> str:
    """The arm on the target's side: the sign of the target's y in the base frame."""
    yaw = float(info["base_yaw_rad"])
    delta = target[:2] - np.asarray(info["base_position"], dtype=np.float64)[:2]
    lateral = -np.sin(yaw) * delta[0] + np.cos(yaw) * delta[1]
    return "left" if lateral >= 0 else "right"


def press(env, info: dict, target: np.ndarray, arm: str) -> dict:
    """Press a closed gripper onto a point from a few offsets, holding after
    each so ToggledOn's 5-frame contact window can close."""
    rotvec = _rotvec_of(info["arms"][arm]["eef_rotation"])
    grips = (+1.0, -1.0) if arm == "left" else (-1.0, +1.0)
    _obs, _r, _t, _tr, info = env.step(BehaviorPoseEnv.hold_action(*grips))
    _report("gripper(close, make a tool)", info)
    # The IK loop converges monotonically while the target is held (measured 2026-09-10:
    # 0.042 m -> 0.031 m on the second identical command), so the press is the same point
    # commanded again and again, with a gripper hold between each so ToggledOn's contact
    # window can close. One approach from above first, so the arm comes down onto the button.
    offsets = [(0.0, 0.0, 0.06)] + [(0.0, 0.0, 0.0)] * 9
    best = float("inf")
    for dx, dy, dz in offsets:
        point = target + [dx, dy, dz]
        _obs, _r, term, trunc, info = env.step(
            BehaviorPoseEnv.arm_action(arm, point, rotvec, grips[0] if arm == "left" else grips[1],
                                       other_gripper=grips[1] if arm == "left" else grips[0]))
        _report(f"move_ee({arm}, press {dx:+.2f} {dy:+.2f} {dz:+.2f})", info)
        best = min(best, float(info["position_error_m"]))
        _obs, _r, term, trunc, info = env.step(BehaviorPoseEnv.hold_action(*grips))
        _report("gripper(hold on the button)", info)
        if info["success"] or term or trunc:
            break
    print(f"  closest the gripper came to the button: {best:.4f} m", flush=True)
    return info


def grasp(env, info: dict, target: np.ndarray, arm: str) -> dict:
    """Reach above an object, descend onto it, close, and lift."""
    rotvec = _rotvec_of(info["arms"][arm]["eef_rotation"])
    open_grip, close_grip = -1.0, +1.0
    other = -1.0
    for label, point, grip in (("above", target + [0.0, 0.0, LIFT_M], open_grip),
                               ("onto", target, open_grip)):
        _obs, _r, term, trunc, info = env.step(
            BehaviorPoseEnv.arm_action(arm, point, rotvec, grip, other_gripper=other))
        _report(f"move_ee({arm}, {label})", info)
        if term or trunc:
            return info
    grips = (close_grip, other) if arm == "left" else (other, close_grip)
    _obs, _r, term, trunc, info = env.step(BehaviorPoseEnv.hold_action(*grips))
    _report("gripper(close)", info)
    _obs, _r, term, trunc, info = env.step(
        BehaviorPoseEnv.arm_action(arm, target + [0.0, 0.0, 2 * LIFT_M], rotvec, close_grip, other_gripper=other))
    _report(f"move_ee({arm}, lift)", info)
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="turning_on_radio", help="the BDDL activity to run")
    ap.add_argument("--split", default="mini", choices=list(B.SPLITS))
    ap.add_argument("--check", action="store_true", help="load, print the scene's facts, and stop")
    ap.add_argument("--behavior-root", default=os.environ.get("OMNIGIBSON_DATA_PATH"))
    args = ap.parse_args()

    t0 = time.time()
    env = es.make("behavior-1k", args.split, behavior_root=args.behavior_root)
    index = _episode_of(env, args.task)
    print(f"[{time.time() - t0:6.1f}s] loading {args.task} (episode {index}) — Isaac boots and the house "
          "loads, which takes minutes", flush=True)
    obs, info = env.reset(options={"episode": index})
    base = env.unwrapped
    world = base.world
    print(f"[{time.time() - t0:6.1f}s] loaded {base.episode.episode_id} in {base.episode.scene.scene_model}",
          flush=True)
    print("  instruction  :", base.episode.instruction)
    print("  BDDL goal    :", info["language"])
    print("  predicates   :", [" ".join(p) for p in info["goal"]["predicates"]])
    print("  tick cap     :", info["tick_cap"], "(2x this task's mean human demo)")
    print("  frames       :", {k: tuple(v.shape) for k, v in obs.items()})
    print("  action slices:", {k: (s.start, s.stop) for k, s in world.action_slices.items()})
    print("  base         :", np.round(info["base_position"], 3).tolist(),
          f"yaw={np.degrees(info['base_yaw_rad']):.0f} deg")
    # The macro protocol re-expresses a world-frame target in the frame the IK controller
    # reads. Feeding it the CURRENT eef pose must come back as the controller's own
    # reference pose — if this drifts, every absolute target is anchored on the wrong link.
    for arm in ("left", "right"):
        mine = world.to_controller_frame(arm, *world.eef_pose(arm))[0]
        ref = world.controller_reference(arm)[0]
        print(f"  frame check {arm:5s}: to_controller_frame {np.round(mine, 4).tolist()} vs "
              f"the controller's own {np.round(ref, 4).tolist()} "
              f"(|d|={float(np.linalg.norm(mine - ref)):.6f} m)")
    for arm in ("left", "right"):
        print(f"  eef {arm:5s}    :", np.round(info["arms"][arm]["eef_position"], 3).tolist(),
              f"gripper={info['arms'][arm]['gripper_open']:.0f}")

    objects = info["objects"]
    names = _scope_names(info)
    print(f"  task objects : {len(objects)} placed, {len(names)} named by the goal")
    targets: list[tuple[str, np.ndarray, str]] = []
    for name in names:
        button = _toggle_link(world, name)
        centre = np.asarray(objects[name]["position"], dtype=np.float64)
        kind = "toggle" if button is not None else "grasp"
        point = button if button is not None else centre
        targets.append((name, point, kind))
        print(f"    {name:45s} {kind:6s} at {np.round(point, 3).tolist()} (centre {np.round(centre, 3).tolist()})")
    if args.check:
        env.close()
        return 0
    if not targets:
        print("no goal object was placed in the scene — nothing to drive to")
        env.close()
        return 1

    _report("reset", info)
    name, point, kind = targets[0]
    print(f"[{time.time() - t0:6.1f}s] target: {name} ({kind}) at {np.round(point, 3).tolist()}", flush=True)
    info = _drive_to(env, info, point)
    arm = _nearer_arm(info, point)
    print(f"  reaching with the {arm} arm", flush=True)
    info = press(env, info, point, arm) if kind == "toggle" else grasp(env, info, point, arm)

    m = info["metrics"]
    print(f"[{time.time() - t0:6.1f}s] done — success={info['success']} q_score={m['q_score']:.3f} "
          f"steps={m['steps_taken']} ticks={m['ticks']}", flush=True)
    print("  goal status  :", info["goal_status"])
    print("  travel (m)   :", {k: round(v, 3) for k, v in info["travel_m"].items()})
    print("  metrics      :", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})
    env.close()
    # Isaac's teardown segfaults after og.clear() during interpreter shutdown; the run is over.
    os._exit(0 if info["success"] or m["q_score"] > 0 else 2)


if __name__ == "__main__":
    raise SystemExit(main())
