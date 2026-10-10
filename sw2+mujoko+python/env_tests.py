#!/usr/bin/env python3
"""
env_tests.py - checks the hand + door environment (door_env.py) before training and prints PASS / FAIL with numbers.

  python env_tests.py                       # all tests on Mujoko/Hand_forearm_v5 + Mujoko/Full_door_v4
  python env_tests.py --quick               # everything except the ~1 min learnability test
  python env_tests.py --only collisions     # one group: info start collisions latch actions obs reward random
                                            #            modes chain doors speed learn
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
    ho = env.geo["door"].get("handle_out", {})
    report("info", "robot works on the side with the main lever", True,
           f"handle surface sticking out: front {ho.get('front')} cm2, back {ho.get('back')} cm2 -> robot on the "
           f"'{ho.get('chosen')}' side (DoorEnv(door_side=...) to choose)")
    gp = env.grasp_point() - np.array(env.data.xipos[env.handle_body])
    report("info", "grasp point = the handle's lever in front of the door", env.lever_found,
           f"{'lever found' if env.lever_found else 'no lever in front of the door - using the centre of mass'}; "
           f"{np.linalg.norm(gp) * 1000:.0f} mm from the handle's centre of mass")
    report("info", "hand has collision shapes", env.n_collision_geoms[0] > 0,
           f"{env.n_collision_geoms[0]} hand geoms collide")
    m = env.model
    per = {}
    for g in range(m.ngeom):
        b = m.body(m.geom_bodyid[g]).name
        if not b.startswith("hand/") and (m.geom_contype[g] or m.geom_conaffinity[g]):
            per[b or "world"] = per.get(b or "world", 0) + 1
    report("info", "door parts with collision shapes", m.body(env.door_body).name in per and
           m.body(env.handle_body).name in per, ", ".join(f"{k}: {v}" for k, v in sorted(per.items())))
    for r in env.geo.get("report", []):
        report("info", "scene fix applied", True, r)
    # MuJoCo's per-body collision bits must match the geoms (the cause of "hand passes through the door")
    if hasattr(m, "body_contype"):
        bad = []
        for b in range(m.nbody):
            gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]
            ct = ca = 0
            for g in gs:
                ct |= int(m.geom_contype[g])
                ca |= int(m.geom_conaffinity[g])
            if (ct, ca) != (int(m.body_contype[b]), int(m.body_conaffinity[b])):
                bad.append(m.body(b).name)
        report("info", "per-body collision bits match the geoms", not bad, f"mismatched bodies: {bad or 'none'}")
    # handle collision boxes cover the visible handle
    hb = env.handle_body
    boxes = [g for g in range(m.ngeom) if m.geom_bodyid[g] == hb and m.geom(g).name.startswith("door/handle_box")]
    if boxes:
        from door_env import fit_boxes
        vis = [g for g in range(m.ngeom) if m.geom_bodyid[g] == hb and int(m.geom_type[g]) == 7]
        rng = np.random.default_rng(1)
        pts = []
        for g in vis:                                        # surface points of the visible mesh (body frame)
            mid = m.geom_dataid[g]
            V = np.array(m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]])
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, m.geom_quat[g])
            pts.append(V[rng.choice(len(V), min(len(V), 2000))] @ R.reshape(3, 3).T + m.geom_pos[g])
        P = np.vstack(pts)
        inside = np.zeros(len(P), bool)
        for g in boxes:
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, m.geom_quat[g])
            loc = (P - m.geom_pos[g]) @ R.reshape(3, 3)
            inside |= np.all(np.abs(loc) <= m.geom_size[g] + 1e-4, axis=1)
        report("info", "handle collision boxes cover the handle", inside.mean() > 0.97,
               f"{len(boxes)} boxes cover {inside.mean() * 100:.1f} % of the handle's surface points")


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
    for task in DoorEnv.TASKS:
        env = DoorEnv(task=task, hand_path=H, door_path=D, randomize=False)
        env.reset(seed=0)
        m, d = env.model, env.data
        peak_h, peak_d = 0.0, 0.0
        for _ in range(100):                                 # 2 s, nobody touches anything
            for _ in range(env.frame_skip):
                mujoco.mj_step(m, d)
            peak_h = max(peak_h, abs(math.degrees(d.qpos[env.qh] - env.h0)))
        lim = 3.0 if task == "door" else 1.0              # 'door' starts unlatched (handle already turned)
        report("latch", f"{task}: door stays still when nothing touches it", peak_h < lim,
               f"door moved {peak_h:.2f} deg in 2 s on its own (limit {lim:.0f}"
               f"{', unlatched' if task == 'door' else ''})")
    env = DoorEnv(task="handle", hand_path=H, door_path=D, randomize=False)
    env.reset(seed=0)
    nrm = env.geo["door"]["normal"]
    peak, turned = 0.0, 0.0
    for _ in range(100):                                     # push hard toward the door
        latched = env._latched
        _, _, _, _, info = env.step(act(env, arm_pos=-nrm))
        turned = max(turned, info["handle_frac"])
        if latched and env._latched:                         # only while the latch is engaged
            peak = max(peak, abs(info["door_deg"]))
    report("latch", "door stays shut while the latch is engaged", peak < 2.0,
           f"door moved {peak:.2f} deg while latched (pushing turned the handle to {turned * 100:.0f} %)")

    worst, cases = 0.0, 0
    for task in ("reach", "handle", "full"):                # hammer the door with random actions
        env = DoorEnv(task=task, hand_path=H, door_path=D, randomize_physics=True)
        rng = np.random.default_rng(1)
        for ep in range(4):
            env.reset(seed=50 + ep)
            for _ in range(250):
                latched = env._latched                      # engaged before this step
                _, _, term, trunc, info = env.step(rng.uniform(-1, 1, env.action_space.shape).astype(np.float32))
                if latched and env._latched:                # still engaged -> the door must not have moved
                    worst = max(worst, abs(info["door_deg"]))
                    cases += 1
                if term or trunc:
                    break
    report("latch", "latch never gives without the handle (random actions)", worst < 2.0,
           f"{cases} latched steps over 12 random episodes, door moved at most {worst:.2f} deg")

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
    away = env.geo["door"]["normal"]
    for ch in ("index", "mrp", "thumb", "opp"):
        env.reset(seed=0)
        for _ in range(15):                                  # into free space first (no lever in the way)
            env.step(act(env, arm_pos=away))
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
            to = env.grasp_point() - np.array(d.xipos[env.palm])
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
           f"door mass {env.model.body_mass[env.door_body]:.1f} kg (nominal {nominal:.1f}), "
           f"latch at {env.unlock_frac * 100:.0f} %")
    fixed = DoorEnv(task="door", hand_path=H, door_path=D, physics={"door_mass": 1.8, "hinge_friction": 3.0})
    p = fixed.reset(seed=0)[1]["physics"]
    report("random", "fixed test door (physics=...) is applied", p["door_mass"] == 1.8 and p["hinge_friction"] == 3.0,
           f"door_mass {p['door_mass']}, hinge_friction {p['hinge_friction']}")
    off = DoorEnv(task="door", hand_path=H, door_path=D)
    report("random", "randomization off -> nominal door", off.reset(seed=0)[1]["physics"] == off.NOMINAL,
           "every value at its nominal")
    mv = DoorEnv(task="reach", hand_path=H, door_path=D)
    _, info0 = mv.reset(seed=0)
    g0 = mv.geo["door"]["grasp"].copy()
    arm0 = mv.data.qpos[mv.free_q:mv.free_q + 3].copy()
    _, info = mv.reset(seed=0, options={"physics": {"door_dx": 0.10, "door_dz": 0.05}})
    moved = np.linalg.norm(mv.geo["door"]["grasp"] - g0)
    arm_moved = np.linalg.norm(mv.data.qpos[mv.free_q:mv.free_q + 3] - arm0)
    report("random", "door_dx / door_dz really move the handle", abs(moved - math.hypot(0.10, 0.05)) < 0.005,
           f"handle moved {moved * 100:.1f} cm (expected {math.hypot(0.10, 0.05) * 100:.1f})")
    report("random", "the robot stays put when the door moves (reach)", arm_moved < 0.005,
           f"robot moved {arm_moved * 1000:.1f} mm; start palm->handle {info0['palm_to_handle'] * 1000:.0f} -> "
           f"{info['palm_to_handle'] * 1000:.0f} mm")


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


def t_modes(H, D):
    print("\n== ACTION MODES / TASKS")
    from door_env import scripted_action
    for mode in ("synergy", "full"):
        for task in DoorEnv.TASKS:
            env = DoorEnv(task=task, hand_path=H, door_path=D, action_mode=mode)
            ok = 0
            for sd in range(3):
                env.reset(seed=sd)
                done = False
                while not done:
                    _, _, term, trunc, info = env.step(scripted_action(env))
                    done = term or trunc
                ok += info["success"]
            report("modes", f"{mode:<7} {task:<6} solvable (scripted)", ok >= 2,
                   f"{ok}/3 episodes, {env.action_space.shape[0]} actions (hand-written policy; 2/3 proves it can be done)")
    env = DoorEnv(task="handle", hand_path=H, door_path=D, action_mode="full")
    for jn in env.joint_names[:: max(1, len(env.joint_names) // 4)]:
        env.reset(seed=0)
        q0 = {n: env.data.qpos[env.hj[n]["qadr"]] for n in env.joint_names}
        k = 6 + env.joint_names.index(jn)
        a = np.zeros(env.action_space.shape, dtype=np.float32)
        a[k] = 1.0
        for _ in range(15):
            env.step(a)
        dq = {n: abs(env.data.qpos[env.hj[n]["qadr"]] - q0[n]) for n in env.joint_names}
        others = max(v for n, v in dq.items() if n != jn)
        report("modes", f"full: joint action moves only {jn.split('__')[-1][:22]}",
               dq[jn] > math.radians(5) and others < math.radians(3),
               f"it moved {math.degrees(dq[jn]):.0f} deg, others max {math.degrees(others):.1f} deg")


def t_chain(H, D):
    print("\n== CHAINING THE MODULAR POLICIES (scripted)")
    from door_env import scripted_action
    env = DoorEnv(task="reach", hand_path=H, door_path=D)
    ok, steps = 0, []
    for sd in range(5):
        env.reset(seed=sd)
        good, n = True, 0
        for stage in ("reach", "handle", "door"):
            if stage != "reach":
                env.switch_task(stage)
            done = False
            while not done:
                _, _, term, trunc, info = env.step(scripted_action(env))
                done = term or trunc
            n += env.steps
            if not info["success"]:
                good = False
                break
        ok += good
        steps.append(n)
    report("chain", "reach -> handle -> door on one door", ok >= 3,
           f"{ok}/5 end-to-end, {np.mean(steps):.0f} steps (hand-written policies)")
    env.reset(seed=0)
    report("chain", "reset() returns to the original task", env.task == "reach" and env.action_space.shape[0] == 6,
           f"task '{env.task}', {env.action_space.shape[0]} actions")
    # save / restore the simulation state
    env.reset(seed=1)
    rng = np.random.default_rng(0)
    acts = [rng.uniform(-1, 1, env.action_space.shape).astype(np.float32) for _ in range(15)]
    for a_ in acts[:5]:
        env.step(a_)
    st = env.get_state()
    for a_ in acts[5:]:
        o1 = env.step(a_)[0]
    env.set_state(st)
    for a_ in acts[5:]:
        o2 = env.step(a_)[0]
    report("chain", "get_state / set_state reproduces the motion", np.allclose(o1, o2, atol=1e-4),
           f"max difference after 10 steps {np.abs(o1 - o2).max():.2e}")


def t_doors(H, D):
    print("\n== TEST DOORS")
    from door_env import door_set
    rng_keys = DoorEnv.PHYSICS_RANGES
    ind = door_set("in_dist")
    inside = all(rng_keys[k][0] - 1e-9 <= v <= rng_keys[k][1] + 1e-9 for _, p in ind for k, v in p.items())
    report("doors", "in-distribution doors are inside the training ranges", inside and len(ind) == 10,
           f"{len(ind)} doors")
    uns = door_set("unseen")
    outside = [any(not (rng_keys[k][0] <= v <= rng_keys[k][1]) for k, v in p.items()) for _, p in uns]
    report("doors", "every unseen door is outside the training ranges", all(outside),
           f"{sum(outside)}/{len(uns)}: " + ", ".join(lbl for lbl, _ in uns))
    report("doors", "the sets are the same every time", door_set("in_dist") == ind, "fixed seed")


def t_learn(H, D):
    print("\n== LEARNABILITY (short ARS run on 'reach')")
    import types
    import ars
    a = types.SimpleNamespace(task="reach", hand=H, door=D, action_mode="synergy", condition="fixed",
                              obs_noise=0.0, seed=0, iters=10, dirs=6, top=3, step=0.02, noise=0.03,
                              eval_every=10, eval_episodes=6, workers=1, out="env_test_ars_tmp")
    t0 = time.time()
    pol = ars.train(a, log=lambda *x: None)
    env = DoorEnv(task="reach", hand_path=H, door_path=D)
    zero = ars.LinearPolicy(pol.mean.size, pol.W.shape[0])

    def score(p):
        ok, closest = 0, []
        for i in range(6):
            obs, _ = env.reset(seed=900 + i)
            done, best = False, 9.0
            while not done:
                obs, _, term, trunc, info = env.step(p.act(obs))
                best = min(best, info["palm_to_handle"])
                done = term or trunc
            ok += info["success"]
            closest.append(best)
        return ok / 6, float(np.mean(closest))
    (s0, c0), (s1, c1) = score(zero), score(pol)
    import shutil
    shutil.rmtree("env_test_ars_tmp", ignore_errors=True)
    report("learn", "a policy learns 'reach' in 10 ARS iterations", c1 < 0.7 * c0,
           f"closest palm->handle {c0 * 1000:.0f} -> {c1 * 1000:.0f} mm, success {s0 * 100:.0f} -> {s1 * 100:.0f} % "
           f"({time.time() - t0:.0f} s; train longer with ars.py / experiment.py for full success)")


GROUPS = dict(info=t_info, start=t_start, collisions=t_collisions, latch=t_latch, actions=t_actions,
              obs=t_obs, reward=t_reward, random=t_random, modes=t_modes, chain=t_chain, doors=t_doors,
              speed=t_speed, learn=t_learn)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hand", default=DEFAULT_HAND)
    ap.add_argument("--door", default=DEFAULT_DOOR)
    ap.add_argument("--only", choices=list(GROUPS), action="append")
    ap.add_argument("--quick", action="store_true", help="skip the slow learnability test")
    args = ap.parse_args()
    for name in (args.only or [g for g in GROUPS if not (args.quick and g == "learn")]):
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
