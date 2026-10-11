#!/usr/bin/env python3
"""
watch.py - watch your TRAINED policies in the MuJoCo viewer.

  python watch.py                                    # the 3 chained policies (random, seed 0) on the nominal door
  python watch.py --doors unseen                     # ... on all 9 unseen test doors, one after another
  python watch.py --test-door turned_door            # ... on one test door
  python watch.py --condition fixed --seed 2         # another trained set
  python watch.py --runs runs_v3                     # runs from another run_all folder
  python watch.py --single runs\\handle_ars_random_design_scratch_s0   # one policy on its own task
  python watch.py --speed 0.5                        # slow motion

In the viewer:  SPACE = pause / continue,  N = next episode,  close the window = quit.
The console prints each stage change (reach -> handle -> door), recoveries and the result.
"""
import argparse
import glob
import os
import time

import mujoco
import mujoco.viewer

from door_env import door_set
from experiment import load_policy, make_env

CONTROL_DT = 0.04          # one environment step = 1/25 s


def find_run(runs, task, cond, mode, seed):
    hits = sorted(d for d in glob.glob(os.path.join(runs, f"{task}_ars_{cond}_{mode}*_s{seed}"))
                  if os.path.exists(os.path.join(d, "config.json")))
    if not hits:
        raise SystemExit(f"no trained '{task}' run found as {runs}/{task}_ars_{cond}_{mode}*_s{seed}\n"
                         f"(see the folders in '{runs}', or give --reach/--handle/--door explicitly)")
    return hits[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs", help="folder with the trained runs (default runs)")
    ap.add_argument("--condition", choices=["fixed", "random"], default="random")
    ap.add_argument("--mode", default="design", help="action mode of the runs (default design)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reach", default=None)
    ap.add_argument("--handle", default=None)
    ap.add_argument("--door", default=None)
    ap.add_argument("--single", default=None, help="one run directory: watch that policy on its own task")
    ap.add_argument("--doors", choices=["nominal", "in_dist", "unseen", "all"], default="nominal")
    ap.add_argument("--test-door", default=None, help="one test door, e.g. turned_door, handle_low, hard_combo")
    ap.add_argument("--episodes", type=int, default=1, help="episodes per door")
    ap.add_argument("--speed", type=float, default=1.0, help="1 = real time, 0.5 = slow motion")
    ap.add_argument("--no-recovery", action="store_true")
    a = ap.parse_args()

    # ---- which policies
    if a.single:
        pol, cfg = load_policy(a.single)
        stages = [cfg["task"]]
        pols = {cfg["task"]: pol}
        print(f"[watch] {a.single}  (task {cfg['task']})")
    else:
        paths = {t: getattr(a, t) or find_run(a.runs, t, a.condition, a.mode, a.seed)
                 for t in ("reach", "handle", "door")}
        loaded = {t: load_policy(p) for t, p in paths.items()}
        pols = {t: v[0] for t, v in loaded.items()}
        cfg = loaded["reach"][1]
        stages = ["reach", "handle", "door"]
        for t, p in paths.items():
            print(f"[watch] {t:<6} <- {p}")

    # ---- which doors
    doors = dict(door_set("all"))
    if a.test_door:
        if a.test_door not in doors:
            raise SystemExit(f"unknown test door '{a.test_door}'; choose from: {', '.join(doors)}")
        todo = [(a.test_door, doors[a.test_door])]
    else:
        todo = list(door_set(a.doors))

    env = make_env(cfg, task=stages[0], randomize_physics=False, start_states=None)
    state = {"pause": False, "next": False}

    def on_key(key):
        if key == 32:                      # space
            state["pause"] = not state["pause"]
            print("[watch] paused" if state["pause"] else "[watch] running")
        elif key in (78, 110):             # N / n
            state["next"] = True

    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=on_key) as viewer:
        viewer.cam.lookat[:] = env.geo["door"]["grasp"]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.4, 135, -15
        total, wins = 0, 0
        for label, phys in todo:
            for ep in range(a.episodes):
                if not viewer.is_running():
                    return
                if env.task != stages[0]:
                    env.switch_task(stages[0])
                obs, _ = env.reset(seed=200_000 + ep, options={"physics": phys})
                viewer.cam.lookat[:] = env.geo["door"]["grasp"]
                stage, in_stage, recov, steps, success = stages[0], 0, 0, 0, False
                print(f"\n[watch] door '{label}', episode {ep + 1}: {stage}", end="", flush=True)
                state["next"] = False
                while viewer.is_running() and steps < 750 and not state["next"]:
                    if state["pause"]:
                        viewer.sync()
                        time.sleep(0.05)
                        continue
                    t0 = time.time()
                    act, _ = pols[stage].predict(obs, deterministic=True)
                    obs, r, term, trunc, info = env.step(act)
                    steps += 1
                    in_stage += 1
                    viewer.sync()
                    nxt = None
                    if info["success"]:
                        if stage == stages[-1]:
                            success = True
                            break
                        nxt = stages[stages.index(stage) + 1]
                    elif len(stages) > 1 and not a.no_recovery and recov < 3:
                        if stage == "handle" and info["palm_to_handle"] > 0.12 and in_stage > 10:
                            nxt, recov = "reach", recov + 1
                        elif stage == "door" and env._latched:
                            nxt, recov = "handle", recov + 1
                    if nxt is None and (in_stage >= 250 or term and not info["success"]):
                        break
                    if nxt:
                        obs, stage, in_stage = env.switch_task(nxt), nxt, 0
                        print(f" -> {stage}", end="", flush=True)
                    time.sleep(max(0.0, CONTROL_DT / max(a.speed, 1e-3) - (time.time() - t0)))
                total += 1
                wins += success
                print(f"   {'SUCCESS' if success else 'failed'}  ({steps * CONTROL_DT:.1f} s, door "
                      f"{info['door_deg']:.0f} deg, handle {info['handle_frac'] * 100:.0f} %"
                      f"{f', {recov} recoveries' if recov else ''})")
                for _ in range(int(1.0 / 0.05)):                      # 1 s pause to see the end state
                    if not viewer.is_running():
                        break
                    viewer.sync()
                    time.sleep(0.05)
        print(f"\n[watch] {wins}/{total} succeeded")
        while viewer.is_running():                                    # keep the window open at the end
            viewer.sync()
            time.sleep(0.05)


if __name__ == "__main__":
    main()
