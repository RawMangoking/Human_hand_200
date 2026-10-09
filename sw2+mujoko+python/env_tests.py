#!/usr/bin/env python3
"""
env_tests.py - checks the hand + door environment (door_env.py) before training and prints PASS / FAIL with numbers.

  python env_tests.py                       # all tests on Mujoko/Hand_forearm_v5 + Mujoko/Full_door_v4
  python env_tests.py --only collisions     # one group: info, start, collisions, latch, actions, obs, reward,
                                            #            random, speed
  python env_tests.py --hand <xml> --door <xml>
A JSON copy of the results is written to env_test_report.json.
"""
import argparse
import json
import math
import sys
import time

import mujoco
import numpy as np

from door_env import DEFAULT_DOOR, DEFAULT_HAND, DoorEnv

RESULTS = []


def report(group, name, ok, detail):
    RESULTS.append(dict(group=group, test=name, ok=bool(ok), detail=detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<46} {detail}")


def contacts(env):
    m, d, hb = env.model, env.data, env.hand_bodies
    self_c, door_c, deepest = 0, 0, 0.0
    for c in d.contact[:d.ncon]:
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if b1 in hb and b2 in hb:
            self_c += 1
        elif (b1 in hb) != (b2 in hb):
            door_c += 1
            deepest = max(deepest, -float(c.dist))
    return self_c, door_c, deepest


def act(env, **channels):
    """Build an action from named channels, e.g. act(env, arm_pos=[0,0,1], mrp=1)."""
    a = np.zeros(env.action_space.shape, dtype=np.float32)
    k = 0
    for name, n in env.action_layout:
        if name in channels:
            a[k:k + n] = channels[name]
        k += n
    return a


def door_faces(env):
    """(front, back) of the door slab along the door normal, measured from the handle centre (m)."""
    from door_setup import geom_points
    m, d = env.model, env.data
    pts = np.vstack([geom_points(m, d, g) for g in range(m.ngeom) if m.geom_bodyid[g] == env.door_body])
    off = (pts - env.geo["door"]["grasp"]) @ env.geo["door"]["normal"]
    return float(off.max()), float(off.min())


def hand_behind_door(env):
    """How far the deepest hand part has gone past the BACK face of the door (m, > 0 = through the door).
    Only hand parts that are in front of the door opening (not beside it) count."""
    from door_setup import geom_points
    m, d = env.model, env.data
    _, back = door_faces(env)
    g = env.geo["door"]
    worst = -1.0
    for gi in range(m.ngeom):
        if m.geom_bodyid[gi] in env.hand_bodies and (m.geom_contype[gi] or m.geom_conaffinity[gi]):
            off = (geom_points(m, d, gi) - g["grasp"]) @ g["normal"]
            worst = max(worst, back - float(off.min()))
    return worst


def palm_side(env):
    """Signed distance of the palm centre from the handle along the door normal (+ = the hand's own side)."""
    d = env.data
    return float((np.array(d.xipos[env.palm]) - env.geo["door"]["grasp"]) @ env.geo["door"]["normal"])


# --------------------------------------------------------------------------- groups
def t_info(H, D):
    print("\n== INFO")
    for task in DoorEnv.TASKS:
        env = DoorEnv(task=task, hand_path=H, door_path=D)
        obs, info = env.reset(seed=0)
        lay = ", ".join(f"{n}({k})" for n, k in env.action_layout)
        report("info", f"{task}: builds", True,
               f"obs {obs.shape[0]}, actions {env.action_space.shape[0]} = {lay}")
    g = env.geo["door"]
    report("info", "geometry", True,
           f"handle at {np.round(g['grasp'], 3).tolist()} m, door normal {np.round(g['normal'], 2).tolist()}, "
           f"push sign {g['push_sign']:+.0f}; collision geoms hand {env.n_collision_geoms[0]}, "
           f"other {env.n_collision_geoms[1]}; control {1 / (env.frame_skip * env.model.opt.timestep):.0f} Hz")
    report("info", "hand has collision shapes", env.n_collision_geoms[0] > 0,
           f"{env.n_collision_geoms[0]} hand geoms collide")


def t_start(H, D):
    print("\n== START POSE (20 random starts per task)")
    for task in DoorEnv.TASKS:
        env = DoorEnv(task=task, hand_path=H, door_path=D)
        pushed, overl, dist, side = [], 0, [], []
        for s in range(20):
            _, info = env.reset(seed=s)
            pushed.append(info["start_pushed_back"])
            overl += len(env.hand_overlaps())
            dist.append(info["palm_to_handle"])
            side.append(palm_side(env))
        report("start", f"{task}: no overlap with the door at start", overl == 0,
               f"overlaps {overl}; palm->handle {min(dist) * 1000:.0f}-{max(dist) * 1000:.0f} mm; "
               f"moved back up to {max(pushed) * 100:.0f} cm")
        report("start", f"{task}: hand on ONE side of the door", min(side) > 0,
               f"palm in front of the handle by {min(side) * 1000:.0f}-{max(side) * 1000:.0f} mm")


def t_collisions(H, D):
    print("\n== COLLISIONS")
    env = DoorEnv(task="handle", hand_path=H, door_path=D, randomize=False)
    env.reset(seed=0)
    worst_self = 0
    for _ in range(40):                                      # full fist, arm still
        env.step(act(env, index=1, mrp=1, thumb=1, opp=1))
        worst_self = max(worst_self, contacts(env)[0])
    report("collisions", "hand may overlap itself (0 self contacts)", worst_self == 0,
           f"max hand<->hand contacts during a fist: {worst_self}")

    env = DoorEnv(task="reach", hand_path=H, door_path=D, randomize=False)
    env.reset(seed=0)
    nrm = env.geo["door"]["normal"]
    touched, deepest, through = 0, 0.0, -1.0
    front, back = door_faces(env)
    for _ in range(120):                                     # drive the arm into the door
        env.step(act(env, arm_pos=-nrm))
        _, dc, dp = contacts(env)
        touched = max(touched, dc)
        deepest = max(deepest, dp)
        through = max(through, hand_behind_door(env))
    report("collisions", "hand is stopped by the door", touched > 0 and deepest < 0.02,
           f"hand<->door contacts {touched}, deepest overlap {deepest * 1000:.1f} mm")
    report("collisions", "no hand part ever gets behind the door", through < 0,
           f"door faces {front * 1000:+.0f} / {back * 1000:+.0f} mm from the handle; "
           f"deepest hand part stayed {-through * 1000:.0f} mm in front of the back face")


def t_latch(H, D):
    print("\n== LATCH / DOOR")
    env = DoorEnv(task="handle", hand_path=H, door_path=D, randomize=False)
    env.reset(seed=0)
    nrm = env.geo["door"]["normal"]
    peak = 0.0
    for _ in range(100):                                     # push on the door without turning the handle
        _, _, _, _, info = env.step(act(env, arm_pos=-nrm))
        peak = max(peak, abs(info["door_deg"]))
    report("latch", "door stays shut while the handle is not turned", peak < 2.0, f"door moved {peak:.2f} deg")

    env = DoorEnv(task="door", hand_path=H, door_path=D, randomize=False)
    env.reset(seed=0)
    push = -nrm * env.door_goal_sign * env.geo["door"]["push_sign"]
    best = 0.0
    for _ in range(150):
        _, _, term, _, info = env.step(act(env, arm_pos=push, grip=-1))
        best = max(best, info["door_deg"])
        if term:
            break
    report("latch", "door opens once the handle is turned", best > 10, f"door reached {best:.1f} deg")


def t_actions(H, D):
    print("\n== ACTION CHANNELS")
    env = DoorEnv(task="reach", hand_path=H, door_path=D, randomize=False)
    for axis in range(3):
        for sgn in (1, -1):
            env.reset(seed=0)
            p0 = env.data.qpos[env.free_q:env.free_q + 3].copy()
            v = np.zeros(3)
            v[axis] = sgn
            for _ in range(10):
                env.step(act(env, arm_pos=v))
            dp = env.data.qpos[env.free_q:env.free_q + 3] - p0
            ok = dp[axis] * sgn > 0.03
            report("actions", f"arm_pos {'xyz'[axis]}{'+' if sgn > 0 else '-'} moves the arm", ok,
                   f"moved {dp[axis] * 100:+.1f} cm along {'xyz'[axis]}")
    for axis in range(3):
        env.reset(seed=0)
        q0 = env.data.qpos[env.free_q + 3:env.free_q + 7].copy()
        v = np.zeros(3)
        v[axis] = 1
        for _ in range(10):
            env.step(act(env, arm_rot=v))
        q1 = env.data.qpos[env.free_q + 3:env.free_q + 7]
        ang = math.degrees(2 * math.acos(min(1.0, abs(float(q0 @ q1)))))
        report("actions", f"arm_rot {'xyz'[axis]} turns the arm", ang > 5, f"turned {ang:.1f} deg")

    env = DoorEnv(task="handle", hand_path=H, door_path=D, randomize=False)
    for ch in ("index", "mrp", "thumb", "opp"):
        env.reset(seed=0)
        names = env.groups[ch]
        q0 = np.array([env.data.qpos[env.hj[n]["qadr"]] for n in names])
        for _ in range(25):
            env.step(act(env, **{ch: 1.0}))
        q1 = np.array([env.data.qpos[env.hj[n]["qadr"]] for n in names])
        moved = np.degrees(np.abs(q1 - q0))
        others = [n for n in env.hj if n not in names and env.hj[n]["cfg"]["role"] == "flex"
                  and n not in env.groups.get("opp", [])]
        q_oth = max((abs(env.data.qpos[env.hj[n]["qadr"]] - env._home(n)) for n in others), default=0)
        report("actions", f"{ch}: moves its {len(names)} joint(s) only",
               len(names) > 0 and moved.min() > 10 and math.degrees(q_oth) < 5,
               f"its joints moved {moved.min():.0f}-{moved.max():.0f} deg, other fingers {math.degrees(q_oth):.1f} deg")
    for i, wn in enumerate(env.wrist):
        env.reset(seed=0)
        q0 = env.data.qpos[env.hj[wn]["qadr"]]
        w = np.zeros(2)
        w[i] = 1
        for _ in range(25):
            env.step(act(env, wrist=w))
        report("actions", f"wrist[{i}] moves {wn.split('__')[-1][:24]}",
               abs(env.data.qpos[env.hj[wn]["qadr"]] - q0) > math.radians(3),
               f"moved {math.degrees(env.data.qpos[env.hj[wn]['qadr']] - q0):+.1f} deg")


def t_obs(H, D):
    print("\n== OBSERVATIONS")
    for task in DoorEnv.TASKS:
        env = DoorEnv(task=task, hand_path=H, door_path=D)
        obs, _ = env.reset(seed=1)
        shape, finite, inside, lo, hi = obs.shape, True, True, np.inf, -np.inf
        rng = np.random.default_rng(0)
        for _ in range(200):
            obs, _, term, trunc, _ = env.step(rng.uniform(-1, 1, env.action_space.shape).astype(np.float32))
            finite &= bool(np.all(np.isfinite(obs)))
            inside &= bool(env.observation_space.contains(obs))
            lo, hi = min(lo, float(obs.min())), max(hi, float(obs.max()))
            if obs.shape != shape:
                finite = False
            if term or trunc:
                env.reset()
        report("obs", f"{task}: finite, fixed size, inside bounds", finite and inside,
               f"200 random steps, values {lo:.2f} .. {hi:.2f}")


def t_reward(H, D):
    print("\n== REWARD")
    env = DoorEnv(task="reach", hand_path=H, door_path=D, randomize=False)
    totals = {}
    for name in ("toward", "away"):
        env.reset(seed=0)
        tot = 0.0
        for _ in range(20):
            d = env.data
            to = np.array(d.xipos[env.handle_body]) - np.array(d.xipos[env.palm])
            to /= np.linalg.norm(to) + 1e-9
            _, r, term, _, _ = env.step(act(env, arm_pos=to if name == "toward" else -to))
            tot += r
            if term:
                break
        totals[name] = tot
    report("reward", "reach: moving toward the handle pays more", totals["toward"] > totals["away"],
           f"toward {totals['toward']:.2f} vs away {totals['away']:.2f}")
    env = DoorEnv(task="reach", hand_path=H, door_path=D)
    o1, _ = env.reset(seed=5)
    o2, _ = env.reset(seed=5)
    o3, _ = env.reset(seed=6)
    report("reward", "reset(seed) is repeatable", np.allclose(o1, o2) and not np.allclose(o1, o3),
           "same seed -> same start, other seed -> different start")


def t_random(H, D):
    print("\n== DOMAIN RANDOMIZATION")
    env = DoorEnv(task="door", hand_path=H, door_path=D, randomize_physics=True)
    samples = [env.reset(seed=s)[1]["physics"] for s in range(30)]
    for k, (lo, hi) in env.physics_ranges.items():
        vals = [p[k] for p in samples]
        ok = min(vals) >= lo - 1e-9 and max(vals) <= hi + 1e-9 and (max(vals) - min(vals)) > 0.3 * (hi - lo)
        report("random", f"{k} varies inside {lo}-{hi}", ok, f"seen {min(vals):.2f} .. {max(vals):.2f}")
    nominal = env._nominal["door_mass"]
    env.reset(seed=3)
    report("random", "values reach the simulation", abs(env.model.body_mass[env.door_body] -
                                                          nominal * env.physics["door_mass"]) < 1e-9,
           f"door mass {env.model.body_mass[env.door_body]:.1f} kg (nominal {nominal:.1f})")
    fixed = DoorEnv(task="door", hand_path=H, door_path=D, physics={"door_mass": 1.8, "hinge_friction": 3.0})
    p = fixed.reset(seed=0)[1]["physics"]
    report("random", "fixed test door (physics=...) is applied", p["door_mass"] == 1.8 and p["hinge_friction"] == 3.0,
           f"{p}")
    off = DoorEnv(task="door", hand_path=H, door_path=D)
    report("random", "randomization off -> nominal door", all(v == 1.0 for v in off.reset(seed=0)[1]["physics"].values()),
           "all multipliers 1.0")


def t_speed(H, D):
    print("\n== SPEED")
    for task in DoorEnv.TASKS:
        env = DoorEnv(task=task, hand_path=H, door_path=D)
        env.reset(seed=0)
        rng = np.random.default_rng(0)
        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < 3.0:
            _, _, term, trunc, _ = env.step(rng.uniform(-1, 1, env.action_space.shape).astype(np.float32))
            n += 1
            if term or trunc:
                env.reset()
        sps = n / (time.perf_counter() - t0)
        report("speed", f"{task}: env steps per second", sps > 20,
               f"{sps:.0f} steps/s ({sps * env.frame_skip:.0f} physics steps/s); 1M steps ~ {1e6 / sps / 3600:.1f} h")


GROUPS = dict(info=t_info, start=t_start, collisions=t_collisions, latch=t_latch, actions=t_actions,
              obs=t_obs, reward=t_reward, random=t_random, speed=t_speed)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hand", default=DEFAULT_HAND)
    ap.add_argument("--door", default=DEFAULT_DOOR)
    ap.add_argument("--only", choices=list(GROUPS), action="append")
    args = ap.parse_args()
    for name in (args.only or GROUPS):
        try:
            GROUPS[name](args.hand, args.door)
        except Exception as e:                                # keep going, report the crash
            report(name, "ran without crashing", False, f"{type(e).__name__}: {e}")
    n_ok = sum(r["ok"] for r in RESULTS)
    print(f"\n{n_ok}/{len(RESULTS)} passed")
    with open("env_test_report.json", "w") as f:
        json.dump(RESULTS, f, indent=2)
    print("results written to env_test_report.json")
    sys.exit(0 if n_ok == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
