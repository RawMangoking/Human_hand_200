#!/usr/bin/env python3
"""
door_scene.py
=============
Scripted demo on the door built by door_setup.py (--handle-hits-frame):

  1. PUSH   the door is pushed open -> the handle hits the frame -> the door stops
  2. TURN   the handle turns until its lever is VERTICAL (angle worked out from the handle geometry)
  3. OPEN   the door swings open while the handle is held
  4. HOLD   pause, then reset and start again (--once to stop after one run)

The push and the handle turn are applied as joint torques (like a hand would), so everything
that stops or frees the door is the real handle <-> frame collision.

Usage
  python door_scene.py <Full_door_v4_sim.xml>
  python door_scene.py <Full_door_v4_sim.xml> --open-dir -1        # swing the other way
  python door_scene.py <Full_door_v4_sim.xml> --once --headless    # no window, prints the result
  python door_scene.py <Full_door_v4_sim.xml> --manual             # drive door + handle with the keyboard
"""
import argparse
import json
import math
import os
import sys
import time

import mujoco
import numpy as np

try:
    import mujoco.viewer as mjviewer
except Exception:
    mjviewer = None


def show_contacts(v, m, forces=False, markers=True):
    """Contact markers sized in real millimetres (MuJoCo scales them by the model's mean size, which a big
    base plate makes huge). Force arrows only on request."""
    ms = max(float(m.stat.meansize), 1e-6)
    v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = bool(markers)
    v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = bool(forces)
    m.vis.scale.contactwidth = 0.008 / ms          # 8 mm dot
    m.vis.scale.contactheight = 0.002 / ms         # 2 mm thick
    m.vis.scale.forcewidth = 0.004 / ms            # 4 mm arrow
    m.vis.map.force = 0.0005                       # 100 N -> 5 cm

def lever_vertical_angle(m, d, jd):
    """Handle angle (joint value) at which the lever (pivot -> handle centre) points straight up or
    down; the one inside the joint range, closest to the rest pose."""
    hb = m.jnt_bodyid[jd]
    adr = m.jnt_qposadr[jd]
    q0 = float(d.qpos[adr])
    a = np.array(d.xaxis[jd])
    r = np.array(d.xipos[hb]) - np.array(d.xanchor[jd])
    r -= a * (r @ a)
    z = np.array([0.0, 0.0, 1.0])
    if abs(a @ z) > 0.95 or np.linalg.norm(r) < 1e-6:
        return None, "the handle axis is vertical (or the lever has no length) - cannot turn it vertical"
    up = math.atan2(float(np.cross(a, r) @ z), float(r @ z))       # extra rotation that points the lever up
    lo, hi = m.jnt_range[jd]
    cands = []
    for extra in (up, up + math.pi, up - math.pi, up + 2 * math.pi, up - 2 * math.pi):
        q = q0 + extra
        if lo - 1e-3 <= q <= hi + 1e-3:
            cands.append((abs(extra), q, "up" if abs(((extra - up + math.pi) % (2 * math.pi)) - math.pi) < 1e-6 else "down"))
    if cands:
        _, q, which = min(cands)
        return float(np.clip(q, lo, hi)), f"lever points {which} at {math.degrees(q - q0):.0f} deg"
    # not reachable: go to the limit that gets closest
    best = max((lo, hi), key=lambda q: abs(math.cos(q - q0 - up)))
    return float(best), (f"vertical is outside the handle range - using the limit "
                         f"{math.degrees(best - q0):.0f} deg (lever {abs(math.degrees(best - q0 - up)) % 180:.0f} deg off vertical)")


class Scene:
    def __init__(self, m, d, cfg, args):
        self.m, self.d, self.a = m, d, args
        self.jh = m.joint(cfg["hinge"]).id
        self.jd = m.joint(cfg["handle"]).id
        self.qh, self.qd = m.jnt_qposadr[self.jh], m.jnt_qposadr[self.jd]
        self.vh, self.vd = m.jnt_dofadr[self.jh], m.jnt_dofadr[self.jd]
        self.h0, self.d0 = float(m.qpos0[self.qh]), float(m.qpos0[self.qd])
        mujoco.mj_forward(m, d)
        self.d_vert, self.vert_msg = lever_vertical_angle(m, d, self.jd)
        lo, hi = m.jnt_range[self.jh]
        lim = hi if args.open_dir > 0 else lo
        if abs(lim - self.h0) < math.radians(5):
            sys.exit(f"the door cannot open in the {'+' if args.open_dir > 0 else '-'} direction: its hinge range is "
                     f"[{math.degrees(lo):.0f}, {math.degrees(hi):.0f}] deg - use --open-dir {-args.open_dir}")
        self.door_goal = self.h0 + (lim - self.h0) * min(1.0, args.open_frac)
        self.reset(announce=False)

    def reset(self, announce=True):
        mujoco.mj_resetData(self.m, self.d)
        mujoco.mj_forward(self.m, self.d)
        self.phase, self.t_phase, self.handle_target = "PUSH", 0.0, self.d0
        if announce:
            self.log(f"PUSH   pushing the door {'+' if self.a.open_dir > 0 else '-'} - the handle should stop it")

    def log(self, msg):
        if not self.a.quiet:
            print(f"[{self.d.time:6.2f}s] {msg}")

    def door_deg(self):
        return math.degrees(self.d.qpos[self.qh] - self.h0)

    def handle_deg(self):
        return math.degrees(self.d.qpos[self.qd] - self.d0)

    def step(self, dt):
        m, d, a = self.m, self.d, self.a
        self.t_phase += dt
        d.qfrc_applied[:] = 0

        # door: push toward the goal at a limited speed (a gentle, steady push)
        def push_door(on):
            if not on:
                return
            v = d.qvel[self.vh]
            v_goal = a.open_dir * a.door_speed
            if (self.door_goal - d.qpos[self.qh]) * a.open_dir <= 0:
                v_goal = 0.0
            d.qfrc_applied[self.vh] = float(np.clip(20.0 * (v_goal - v), -a.push, a.push))

        # handle: PD toward a target that moves at a limited speed
        def hold_handle(target):
            step = math.radians(a.turn_speed) * dt
            self.handle_target += float(np.clip(target - self.handle_target, -step, step))
            e = self.handle_target - d.qpos[self.qd]
            # PD + compensation for the handle's return spring and its weight (like a real hand holding it)
            spring = m.jnt_stiffness[self.jd] * (d.qpos[self.qd] - m.qpos_spring[self.qd])
            d.qfrc_applied[self.vd] = 25.0 * e - 0.8 * d.qvel[self.vd] + spring + d.qfrc_bias[self.vd]

        if self.phase.startswith("WAIT"):
            nxt = self.phase.split(":")[1]
            if nxt in ("OPEN", "HOLD"):
                hold_handle(self.d_vert if self.d_vert is not None else self.d0)
            if self.t_phase > a.pause:
                self.phase, self.t_phase = nxt, 0.0
                if nxt == "TURN":
                    self.log(f"TURN   turning the handle until the lever is vertical ({self.vert_msg})")
                elif nxt == "OPEN":
                    self.log("OPEN   pushing the door open")
            return True
        if self.phase == "PUSH":
            push_door(True)
            if self.t_phase > a.push_time:
                hb = m.jnt_bodyid[self.jd]
                who = sorted({f"{m.geom(c.geom1 if m.geom_bodyid[c.geom2] == hb else c.geom2).name or 'frame'}"
                              for c in d.contact[:d.ncon] if hb in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2])})
                still = abs(d.qvel[self.vh]) < 0.05
                blocked = bool(who) and still
                verdict = ("STOPPED by the handle hitting the frame" if blocked else
                           "NOT blocked!" if not still or not who else "stopped")
                self.log(f"       door at {self.door_deg():.2f} deg (handle {self.handle_deg():.1f} deg) -> {verdict}"
                         + (f" - handle touching: {', '.join(who)}" if who else " - handle touches nothing"))
                if not blocked:
                    self.log("       -> the handle did not stop the door: run door_setup.py with --handle-hits-frame "
                             "and check its [handle-frame] line")
                if self.d_vert is None:
                    self.log("TURN   " + self.vert_msg)
                    self.phase = "HOLD"
                else:
                    self.phase, self.t_phase = "WAIT:TURN", 0.0
        elif self.phase == "TURN":
            hold_handle(self.d_vert)
            push_door(a.push_while_turning)
            if abs(d.qpos[self.qd] - self.d_vert) < math.radians(2) and self.t_phase > 0.3:
                self.log(f"       handle vertical ({self.handle_deg():.1f} deg), door at {self.door_deg():.2f} deg")
                self.phase, self.t_phase = "WAIT:OPEN", 0.0
            elif self.t_phase > 5:
                hb = m.jnt_bodyid[self.jd]
                who = sorted({m.geom(c.geom1 if m.geom_bodyid[c.geom2] == hb else c.geom2).name or "?"
                              for c in d.contact[:d.ncon] if hb in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2])})
                self.log(f"       handle stuck at {self.handle_deg():.1f} deg"
                         + (f" - blocked by the frame ({', '.join(who)}): its turning path crosses the frame"
                            if who else " - check the handle range"))
                self.phase, self.t_phase = "OPEN", 0.0
        elif self.phase == "OPEN":
            hold_handle(self.d_vert)
            push_door(True)
            if (self.door_goal - d.qpos[self.qh]) * a.open_dir < math.radians(2) or self.t_phase > 12:
                self.log(f"       door open at {self.door_deg():.1f} deg")
                self.phase, self.t_phase = "HOLD", 0.0
        elif self.phase == "HOLD":
            hold_handle(self.d_vert if self.d_vert is not None else self.d0)
            if self.t_phase > a.hold_time:
                if a.once:
                    return False
                self.reset()
        return True


# --------------------------------------------------------------------------- manual control
class Manual:
    """Hold keys to push the door / turn the handle. Input is abstract (sets of action names)."""
    def __init__(self, m, d, sc, push, turn_speed):
        self.m, self.d, self.sc = m, d, sc
        self.push, self.turn = push, math.radians(turn_speed)
        self.target = sc.d0                                    # handle target (held where you leave it)
        self.hold = False
        lo, hi = m.jnt_range[sc.jd]
        # D always turns the handle toward its open side (works for clockwise and anticlockwise handles)
        self.open_sign = 1.0 if (hi - sc.d0) >= (sc.d0 - lo) else -1.0

    def apply(self, held, once, dt):
        m, d, sc = self.m, self.d, self.sc
        if "reset" in once:
            mujoco.mj_resetData(m, d)
            mujoco.mj_forward(m, d)
            self.target, self.hold = sc.d0, False
        if "release" in once:
            self.target, self.hold = sc.d0, False
        d.qfrc_applied[:] = 0
        # door: push while a key is held (speed-limited so it moves smoothly)
        want = (1 if "door_open" in held else 0) - (1 if "door_close" in held else 0)
        if want:
            v = d.qvel[sc.vh]
            d.qfrc_applied[sc.vh] = float(np.clip(20.0 * (want * 0.5 - v), -self.push, self.push))
        # handle: turn while held, stays where you leave it; released -> spring takes it back
        turn = (1 if "handle_up" in held else 0) - (1 if "handle_down" in held else 0)
        if turn:
            self.hold = True
            lo, hi = m.jnt_range[sc.jd]
            self.target = float(np.clip(self.target + self.open_sign * turn * self.turn * dt, lo, hi))
        if self.hold:
            e = self.target - d.qpos[sc.qd]
            spring = m.jnt_stiffness[sc.jd] * (d.qpos[sc.qd] - m.qpos_spring[sc.qd])
            d.qfrc_applied[sc.vd] = 25.0 * e - 0.8 * d.qvel[sc.vd] + spring + d.qfrc_bias[sc.vd]

    def touching(self):
        m, d = self.m, self.d
        hb = m.jnt_bodyid[self.sc.jd]
        fb = m.body_parentid[m.jnt_bodyid[self.sc.jh]]
        hits = sorted({m.geom(c.geom1 if m.geom_bodyid[c.geom2] == hb else c.geom2).name or "frame"
                       for c in d.contact[:d.ncon]
                       if {m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]} == {hb, fb}})
        return hits

    def status(self):
        hits = self.touching()
        return [f"door   {self.sc.door_deg():7.1f} deg",
                f"handle {self.sc.handle_deg():7.1f} deg   ({'held' if self.hold else 'free, spring returns it'})",
                "handle touching the frame: " + (", ".join(hits) if hits else "no")]


class DoorKeys:
    HELP = ["CLICK THIS WINDOW, then hold:",
            "W / Up      push the door open        S / Down   push it closed",
            "D / Right   turn the handle open      A / Left   turn it back",
            "Space       let go of the handle      R reset     Esc quit"]

    def __init__(self):
        try:
            import pygame
        except ImportError:
            sys.exit("manual mode needs pygame:  python -m pip install pygame")
        self.pg = pg = pygame
        pg.init()
        self.screen = pg.display.set_mode((620, 230))
        pg.display.set_caption("Door manual control  (click here first)")
        self.font = pg.font.SysFont("consolas", 17)
        K = lambda n: getattr(pg, n, None)
        hold = {"K_w": "door_open", "K_UP": "door_open", "K_s": "door_close", "K_DOWN": "door_close",
                "K_d": "handle_up", "K_RIGHT": "handle_up", "K_a": "handle_down", "K_LEFT": "handle_down"}
        once = {"K_SPACE": "release", "K_r": "reset", "K_ESCAPE": "quit"}
        self.hold = {K(k): v for k, v in hold.items() if K(k) is not None}
        self.once = {K(k): v for k, v in once.items() if K(k) is not None}

    def poll(self):
        pg = self.pg
        once = set()
        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                once.add("quit")
            elif ev.type == pg.KEYDOWN and ev.key in self.once:
                once.add(self.once[ev.key])
        keys = pg.key.get_pressed()
        return {a for k, a in self.hold.items() if keys[k]}, once

    def draw(self, status):
        pg = self.pg
        self.screen.fill((24, 24, 28))
        for i, line in enumerate(self.HELP + [""] + status):
            hit = line.startswith("handle touching the frame:") and not line.endswith(": no")
            col = (255, 110, 90) if hit else (120, 220, 140) if i > len(self.HELP) else (220, 220, 220)
            self.screen.blit(self.font.render(line, True, col), (12, 10 + 22 * i))
        pg.display.flip()

    def close(self):
        self.pg.quit()


def run_manual(m, d, sc, args):
    man = Manual(m, d, sc, args.push, args.turn_speed)
    keys = DoorKeys()
    print("[manual] click the small 'Door manual control' window, then hold the keys shown in it")
    last_touch = None
    with mjviewer.launch_passive(m, d) as v:
        with v.lock():
            show_contacts(v, m, args.show_forces, not args.no_markers)
        wall0, sim0 = time.perf_counter(), d.time
        while v.is_running():
            now = time.perf_counter()
            held, once = keys.poll()
            if "quit" in once:
                break
            with v.lock():
                if "reset" in once:
                    man.apply(held, once, 0.0)
                    wall0, sim0 = now, d.time
                n = 0
                while d.time - sim0 < now - wall0 and n < 200:
                    man.apply(held, set(), m.opt.timestep)
                    mujoco.mj_step(m, d)
                    n += 1
                if n == 200:
                    wall0, sim0 = now, d.time
                touch = man.touching()
            if bool(touch) != last_touch:
                last_touch = bool(touch)
                print(f"[manual] door {sc.door_deg():6.1f} deg, handle {sc.handle_deg():6.1f} deg -> "
                      + (f"handle HITS the frame ({', '.join(touch)})" if touch else "handle clear of the frame"))
            keys.draw(man.status())
            v.sync()
            time.sleep(max(0.0, 1 / 90 - (time.perf_counter() - now)))
    keys.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="the *_sim.xml written by door_setup.py")
    ap.add_argument("--open-dir", type=int, choices=[1, -1], default=1, help="which way the door opens (+1 / -1)")
    ap.add_argument("--open-frac", type=float, default=0.9, help="open to this fraction of the hinge limit (0.9)")
    ap.add_argument("--push", type=float, default=15.0, help="max push on the door, N*m (default 15)")
    ap.add_argument("--door-speed", type=float, default=0.4, help="door opening speed, rad/s (default 0.4)")
    ap.add_argument("--turn-speed", type=float, default=45.0, help="handle turning speed, deg/s (default 45)")
    ap.add_argument("--push-time", type=float, default=2.0, help="seconds of pushing against the handle (default 2)")
    ap.add_argument("--pause", type=float, default=1.0, help="pause between the stages, s (default 1)")
    ap.add_argument("--hold-time", type=float, default=2.0, help="pause when open before restarting (default 2)")
    ap.add_argument("--push-while-turning", action="store_true", help="keep pushing the door while the handle turns")
    ap.add_argument("--once", action="store_true", help="run the sequence once")
    ap.add_argument("--show-forces", action="store_true", help="also draw contact force arrows (small)")
    ap.add_argument("--no-markers", action="store_true", help="no contact markers at all")
    ap.add_argument("--manual", action="store_true",
                    help="drive the door and handle yourself with the keyboard (small extra window)")
    ap.add_argument("--headless", action="store_true", help="no window (with --once: prints the result)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    path = os.path.abspath(args.model)
    cfg_path = os.path.join(os.path.dirname(path), "door_config.json")
    if not os.path.exists(cfg_path):
        sys.exit(f"no door_config.json next to the model - run door_setup.py first")
    cfg = json.load(open(cfg_path))
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    if not cfg.get("physical_latch"):
        print("[note] this door was set up without --handle-hits-frame; the handle may not stop it")
    sc = Scene(m, d, cfg, args)

    if args.manual and not args.headless:
        if mjviewer is None:
            sys.exit("mujoco.viewer unavailable")
        run_manual(m, d, sc, args)
        return

    # does anything move on its own (an overlap being shoved apart)? 0.5 s with no forces
    mujoco.mj_forward(m, d)
    at_start = sorted({f"{m.body(m.geom_bodyid[c.geom1]).name}/{m.geom(c.geom1).name or 'geom ' + str(c.geom1)} <-> "
                       f"{m.body(m.geom_bodyid[c.geom2]).name}/{m.geom(c.geom2).name or 'geom ' + str(c.geom2)}"
                       f" ({-c.dist * 1000:.1f} mm)" for c in d.contact[:d.ncon] if c.dist < -0.0005})
    t = 0.0
    while t < 0.5:
        mujoco.mj_step(m, d)
        t += m.opt.timestep
    moved = math.degrees(abs(d.qpos[sc.qh] - sc.h0))
    if moved > 1.0:
        pairs = sorted({f"{m.body(m.geom_bodyid[c.geom1]).name}/{m.geom(c.geom1).name or c.geom1} <-> "
                        f"{m.body(m.geom_bodyid[c.geom2]).name}/{m.geom(c.geom2).name or c.geom2}"
                        for c in d.contact[:d.ncon]})
        print(f"[WARNING] with NO push the door moved {moved:.1f} deg by itself in 0.5 s - parts overlap at rest.")
        print("          overlapping at the start: " + ("; ".join(at_start) if at_start else "none"))
        print("          contacts now: " + ("; ".join(pairs) if pairs else "none"))
    sc.reset()

    if args.headless:
        t_end = 60.0
        while d.time < t_end and sc.step(m.opt.timestep):
            mujoco.mj_step(m, d)
            if not np.all(np.isfinite(d.qpos)):
                sys.exit("simulation went unstable")
        return

    if mjviewer is None:
        sys.exit("mujoco.viewer unavailable")
    with mjviewer.launch_passive(m, d) as v:
        with v.lock():                                   # show where the handle hits the frame
            show_contacts(v, m, args.show_forces, not args.no_markers)
        wall0, sim0 = time.perf_counter(), d.time
        while v.is_running():
            now = time.perf_counter()
            if d.time < sim0:                             # reset happened
                wall0, sim0 = now, d.time
            n = 0
            with v.lock():
                while d.time - sim0 < now - wall0 and n < 200:
                    if not sc.step(m.opt.timestep):
                        return
                    mujoco.mj_step(m, d)
                    n += 1
                if d.time < sim0:
                    wall0, sim0 = time.perf_counter(), d.time
            v.sync()
            time.sleep(max(0.0, 1 / 90 - (time.perf_counter() - now)))


if __name__ == "__main__":
    main()
