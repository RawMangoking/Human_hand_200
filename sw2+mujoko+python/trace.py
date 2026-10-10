#!/usr/bin/env python3
"""
trace.py - step-by-step log of the hand-written policy on YOUR models, to see why a task fails.

  python trace.py --task handle                 # every 5th step: palm-lever distance, handle %, door, latch,
  python trace.py --task door --every 2         #   whether the arm is blocked, and which hand part touches
  python trace.py --task full --seed 1          #   which door part
  python trace.py --task handle --view          # same, in the viewer (slowed down)
Writes trace_<task>.txt too.
"""
import argparse
import time

import mujoco
import numpy as np

from door_env import DEFAULT_DOOR, DEFAULT_HAND, DoorEnv, pick_scripted_variant, scripted_action


def short(name):
    return name.split("/")[-1].split("__")[-1][:22]


def contacts(env):
    m, d = env.model, env.data
    out = set()
    for c in d.contact[:d.ncon]:
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if (b1 in env.hand_bodies) != (b2 in env.hand_bodies):
            hb, ob = (b1, b2) if b1 in env.hand_bodies else (b2, b1)
            g = c.geom1 if m.geom_bodyid[c.geom1] == ob else c.geom2
            out.add(f"{short(m.body(hb).name)}->{short(m.body(ob).name)}"
                    + (f"[{m.geom(g).name.split('/')[-1]}]" if m.geom(g).name else ""))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=list(DoorEnv.TASKS), default="handle")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--hand", default=DEFAULT_HAND)
    ap.add_argument("--door", default=DEFAULT_DOOR)
    a = ap.parse_args()
    env = DoorEnv(task=a.task, hand_path=a.hand, door_path=a.door)
    sc = pick_scripted_variant(env)
    print(f"hand-written handle strategy: tried {sc} -> using '{env.scripted_variant}'")
    obs, info = env.reset(seed=a.seed)
    d = env.data
    lines = [f"task {a.task}, seed {a.seed}: lever at {np.round(env.grasp_point(), 3).tolist()}, "
             f"handle opens toward {np.degrees(env.handle_open - env.d0):+.0f} deg, latch releases at "
             f"{env.unlock_frac * 100:.0f} %, start palm->lever {info['palm_to_handle'] * 1000:.0f} mm",
             f"{'step':>4} {'stage':>5} {'palm-lever':>10} {'handle':>7} {'door':>6} {'latch':>6} {'arm lag':>8}  "
             f"action (arm xyz | fingers)            contacts hand->door"]
    viewer = None
    if a.view:
        import mujoco.viewer as mv
        viewer = mv.launch_passive(env.model, d)
        viewer.cam.lookat[:] = env.grasp_point()
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.0, 135, -15
    done, k = False, 0
    while not done:
        act = scripted_action(env)
        obs, r, term, trunc, info = env.step(act)
        done = term or trunc
        lag = np.linalg.norm(d.mocap_pos[env.mid] - d.qpos[env.free_q:env.free_q + 3])
        if k % a.every == 0 or done:
            lines.append(f"{k:4d} {env._stage:5d} {info['palm_to_handle'] * 1000:8.0f}mm {info['handle_frac'] * 100:6.0f}%"
                         f" {info['door_deg']:5.1f}d {'LOCK' if env._latched else 'free':>6} {lag * 100:6.1f}cm  "
                         f"[{' '.join(f'{x:+.1f}' for x in act[:3])} | {' '.join(f'{x:+.1f}' for x in act[6:])}]"
                         f"  {', '.join(contacts(env)) or '-'}")
            print(lines[-1])
        if viewer:
            viewer.sync()
            time.sleep(0.06)
            if not viewer.is_running():
                break
        k += 1
    lines.append(f"RESULT: {'SUCCESS' if info['success'] else 'FAIL'} after {env.steps} steps; handle "
                 f"{info['handle_frac'] * 100:.0f} %, door {info['door_deg']:.1f} deg  "
                 f"(arm lag near 2 cm = the arm is pushing against something)")
    print(lines[-1])
    open(f"trace_{a.task}.txt", "w").write("\n".join(lines) + "\n")
    print(f"written trace_{a.task}.txt")


if __name__ == "__main__":
    main()
