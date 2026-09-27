#!/usr/bin/env python3
"""
hand_controller.py
==================
Drive the actuated hand (from add_actuators.py) with a gamepad in the MuJoCo viewer.

  python hand_controller.py robot_actuated.xml            # gamepad control
  python hand_controller.py robot_actuated.xml --sweep    # move every joint one at a time (check axes/limits)
  python hand_controller.py robot_actuated.xml --demo     # cycle open / fist / pinch / point
  python hand_controller.py robot_actuated.xml --pose 45  # hold every finger/thumb joint at 45 deg
  python hand_controller.py robot_actuated.xml --demo --headless 10   # no window, prints tracking error

No gamepad? The viewer still opens: use the "Control" sliders in the right-hand panel.

Gamepad (Xbox names; PlayStation / Switch pads are remapped by SDL automatically)
  SYNERGY mode (default)
    Right trigger ........ close index, middle, ring, pinky
    Left trigger ......... close thumb
    Left stick Y ......... wrist flexion (tilt front/back)      - rate control, stays where you leave it
    Left stick X ......... wrist deviation (tilt left/right), or twist if there is no deviation axis
    D-pad left/right ..... wrist twist (when the wrist has flexion + deviation + twist)
    Right stick X ........ thumb / finger lateral axes added with --add-axis
    A open | B fist | X pinch | Y point   (presets are "held"; triggers close further on top)
    RB hold current grip   LB release hold   R-stick click: centre wrist
  JOINT mode (Start toggles)
    D-pad left/right ..... choose group (wrist, thumb, index, ...)
    D-pad up/down ........ choose joint in that group
    Right stick Y ........ move the selected joint
  Back: reset everything
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
except Exception:          # no GUI libs available: --headless still works
    mjviewer = None

DEADZONE = 0.12
WRIST_RATE = 1.5      # rad/s at full stick
JOINT_RATE = 1.5      # rad/s at full stick (joint mode)
MAX_CTRL_RATE = 6.0   # rad/s - smooths preset jumps so the sim never gets a step input

PRESETS = {           # curl 0 = open, 1 = fully closed ; "default" covers groups not listed
    "open":  dict(default=0.0),
    "fist":  dict(default=1.0, thumb=0.9),
    "pinch": dict(default=0.0, thumb=0.6, index=0.6),
    "point": dict(default=1.0, thumb=0.8, index=0.0),
}
PRESET_BUTTONS = {"a": "open", "b": "fist", "x": "pinch", "y": "point"}


# --------------------------------------------------------------------------- gamepad
class Gamepad:
    """Normalised pad: sticks in [-1,1] (up = -1 on Y), triggers in [0,1], button names Xbox-style."""
    BUTTONS = ["a", "b", "x", "y", "back", "start", "leftstick", "rightstick",
               "leftshoulder", "rightshoulder", "dpad_up", "dpad_down", "dpad_left", "dpad_right"]

    def __init__(self):
        self.kind, self.dev, self.name = None, None, ""
        try:
            import pygame
        except ImportError:
            print("[pad] pygame not installed (pip install pygame) - gamepad disabled")
            return
        self.pg = pygame
        pygame.init()
        try:                                   # SDL GameController: same layout for every brand
            from pygame._sdl2 import controller as sdlc
            sdlc.init()
            for i in range(sdlc.get_count()):
                if sdlc.is_controller(i):
                    self.dev = sdlc.Controller(i)
                    self.kind = "sdl"
                    self.name = getattr(self.dev, "name", "gamepad")
                    break
        except Exception:
            self.kind = None
        if self.kind is None:                  # raw joystick fallback (Xbox layout on Windows)
            pygame.joystick.init()
            if pygame.joystick.get_count():
                self.dev = pygame.joystick.Joystick(0)
                self.dev.init()
                self.kind = "joy"
                self.name = self.dev.get_name()
        self.trig_seen = [False, False]
        if self.kind:
            print(f"[pad] using '{self.name}' ({self.kind})")

    @staticmethod
    def _dz(v):
        return 0.0 if abs(v) < DEADZONE else (v - math.copysign(DEADZONE, v)) / (1 - DEADZONE)

    def read(self):
        pg = self.pg
        pg.event.pump()
        s = dict(lx=0.0, ly=0.0, rx=0.0, ry=0.0, lt=0.0, rt=0.0, buttons=set())
        if self.kind == "sdl":
            ax = lambda c: self.dev.get_axis(getattr(pg, c)) / 32767.0
            s.update(lx=ax("CONTROLLER_AXIS_LEFTX"), ly=ax("CONTROLLER_AXIS_LEFTY"),
                     rx=ax("CONTROLLER_AXIS_RIGHTX"), ry=ax("CONTROLLER_AXIS_RIGHTY"),
                     lt=max(0.0, ax("CONTROLLER_AXIS_TRIGGERLEFT")),
                     rt=max(0.0, ax("CONTROLLER_AXIS_TRIGGERRIGHT")))
            for b in self.BUTTONS:
                if self.dev.get_button(getattr(pg, "CONTROLLER_BUTTON_" + b.upper())):
                    s["buttons"].add(b)
        elif self.kind == "joy":
            d = self.dev
            get = lambda i: d.get_axis(i) if i < d.get_numaxes() else 0.0
            s.update(lx=get(0), ly=get(1), rx=get(2), ry=get(3))
            for k, (key, i) in enumerate((("lt", 4), ("rt", 5))):
                v = get(i)
                if v != 0.0:
                    self.trig_seen[k] = True            # some drivers report 0 until first press
                s[key] = (v + 1) / 2 if self.trig_seen[k] else 0.0
            names = ["a", "b", "x", "y", "leftshoulder", "rightshoulder", "back", "start", "leftstick", "rightstick"]
            for i, b in enumerate(names):
                if i < d.get_numbuttons() and d.get_button(i):
                    s["buttons"].add(b)
            if d.get_numhats():
                hx, hy = d.get_hat(0)
                if hx < 0: s["buttons"].add("dpad_left")
                if hx > 0: s["buttons"].add("dpad_right")
                if hy > 0: s["buttons"].add("dpad_up")
                if hy < 0: s["buttons"].add("dpad_down")
        for k in ("lx", "ly", "rx", "ry"):
            s[k] = self._dz(s[k])
        return s


# --------------------------------------------------------------------------- hand model
class Hand:
    def __init__(self, m, cfg):
        self.m, self.J = m, cfg["joints"]
        self.names = list(self.J)
        self.fingers = cfg["fingers"]
        self.wrist = cfg.get("wrist", [])
        self.wrist_axes = cfg.get("wrist_axes", {})
        self.spread = cfg.get("spread", [])
        self.w = cfg.get("stage_weights", [1.0])
        self.act, self.qadr = {}, {}
        for n in self.names:
            a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, self.J[n]["actuator"])
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
            if a < 0 or j < 0:
                sys.exit(f"config/model mismatch for '{n}' - rerun add_actuators.py")
            self.act[n], self.qadr[n] = a, m.jnt_qposadr[j]
        self.target = {n: self.home(n) for n in self.names}

    def home(self, n):
        j = self.J[n]
        return j["open"] if j["role"] == "flex" else j["neutral"]

    def clamp(self, n, v):
        lo, hi = self.J[n]["range"]
        return min(max(v, lo), hi)

    def set_curl(self, group, c):
        for k, n in enumerate(self.fingers.get(group, [])):
            ck = min(max(c * self.w[min(k, len(self.w) - 1)], 0.0), 1.0)
            j = self.J[n]
            self.target[n] = j["open"] + (j["closed"] - j["open"]) * ck

    def set_bipolar(self, n, x):
        j = self.J[n]
        lo, hi = j["range"]
        nu = j["neutral"]
        x *= j.get("sign", 1)
        self.target[n] = nu + x * (hi - nu) if x >= 0 else nu + x * (nu - lo)

    def reset_targets(self):
        self.target = {n: self.home(n) for n in self.names}

    def apply(self, d, dt):
        step = MAX_CTRL_RATE * dt
        for n in self.names:
            a = self.act[n]
            d.ctrl[a] += float(np.clip(self.target[n] - d.ctrl[a], -step, step))


# --------------------------------------------------------------------------- behaviours
class PadLogic:
    def __init__(self, hand):
        self.h = hand
        self.mode = "synergy"
        self.prev = set()
        self.latch = {g: 0.0 for g in hand.fingers}
        self.groups = (["wrist"] if hand.wrist else []) + list(hand.fingers) + (["spread"] if hand.spread else [])
        self.gi, self.ji = 0, 0
        print("[pad] SYNERGY mode  (Start = switch to JOINT mode)")

    def _group_joints(self, g):
        return self.h.wrist if g == "wrist" else self.h.spread if g == "spread" else self.h.fingers[g]

    def _preset(self, name):
        p = PRESETS[name]
        self.latch = {g: p.get(g, p["default"]) for g in self.h.fingers}
        if self.mode == "joint":
            for g, c in self.latch.items():
                self.h.set_curl(g, c)
        print(f"[pad] preset: {name}")

    def _announce(self):
        g = self.groups[self.gi]
        print(f"[pad] selected {g} / {self._group_joints(g)[self.ji]}")

    def update(self, s, dt):
        h = self.h
        pressed = s["buttons"] - self.prev
        self.prev = s["buttons"]

        if "start" in pressed:
            self.mode = "joint" if self.mode == "synergy" else "synergy"
            print(f"[pad] {self.mode.upper()} mode")
            if self.mode == "joint":
                self._announce()
        if "back" in pressed:
            self.latch = {g: 0.0 for g in h.fingers}
            h.reset_targets()
            print("[pad] reset")
        if "rightstick" in pressed:
            for n in h.wrist:
                h.target[n] = h.home(n)
        for b, preset in PRESET_BUTTONS.items():
            if b in pressed:
                self._preset(preset)
        if "rightshoulder" in pressed:
            self.latch = {g: max(self.latch[g], s["lt"] if g == "thumb" else s["rt"]) for g in h.fingers}
            print("[pad] grip held")
        if "leftshoulder" in pressed:
            self.latch = {g: 0.0 for g in h.fingers}
            print("[pad] hold released")

        # wrist: rate control (it stays where you leave it)
        wa = h.wrist_axes
        if wa:
            side = "deviation" if "deviation" in wa else "twist"
            moves = [(wa.get("flex"), -s["ly"]), (wa.get(side), s["lx"])]
            if side == "deviation" and "twist" in wa and self.mode == "synergy":
                held = s["buttons"]
                moves.append((wa["twist"], (1.0 if "dpad_right" in held else 0.0)
                              - (1.0 if "dpad_left" in held else 0.0)))
        else:
            moves = list(zip(h.wrist[:2], (s["lx"], -s["ly"])))
        for n, val in moves:
            if n:
                h.target[n] = h.clamp(n, h.target[n] + WRIST_RATE * dt * val)

        if self.mode == "synergy":
            for g in h.fingers:
                h.set_curl(g, max(self.latch[g], s["lt"] if g == "thumb" else s["rt"]))
            for n in h.spread:
                h.set_bipolar(n, s["rx"])
        else:
            if "dpad_right" in pressed or "dpad_left" in pressed:
                self.gi = (self.gi + (1 if "dpad_right" in pressed else -1)) % len(self.groups)
                self.ji = 0
                self._announce()
            js = self._group_joints(self.groups[self.gi])
            if "dpad_up" in pressed or "dpad_down" in pressed:
                self.ji = (self.ji + (1 if "dpad_down" in pressed else -1)) % len(js)
                self._announce()
            n = js[self.ji]
            h.target[n] = h.clamp(n, h.target[n] + JOINT_RATE * dt * (-s["ry"]))


class Sweep:
    """Moves each joint through its WHOLE range (home -> upper limit -> lower limit -> home),
    one joint after another, and prints the limits it is going to."""
    def __init__(self, hand, period=4.0):
        self.h, self.T, self.t, self.cur = hand, period, 0.0, -1

    def update(self, dt):
        h = self.h
        self.t += dt
        k = int(self.t // self.T) % len(h.names)
        n = h.names[k]
        j = h.J[n]
        lo, hi = j["range"]
        if k != self.cur:
            self.cur = k
            h.reset_targets()
            print(f"[sweep] {n:<40} {j['group']:<7} {j['role']:<6} "
                  f"range [{math.degrees(lo):6.1f}, {math.degrees(hi):6.1f}] deg")
        home = h.home(n)
        s = math.sin(2 * math.pi * (self.t % self.T) / self.T)      # 0 -> +1 -> 0 -> -1 -> 0
        h.target[n] = home + s * (hi - home) if s >= 0 else home + s * (home - lo)


class Demo:
    def __init__(self, hand, period=2.5):
        self.h, self.T, self.t, self.cur = hand, period, 0.0, None
        self.order = ["open", "fist", "open", "pinch", "open", "point"]

    def update(self, dt):
        self.t += dt
        name = self.order[int(self.t // self.T) % len(self.order)]
        if name != self.cur:
            self.cur = name
            print(f"[demo] {name}")
        p = PRESETS[name]
        for g in self.h.fingers:
            self.h.set_curl(g, p.get(g, p["default"]))
        for i, n in enumerate(self.h.wrist[:2]):
            self.h.set_bipolar(n, 0.4 * math.sin(0.6 * self.t + i * 1.3))


# --------------------------------------------------------------------------- main
def main():
    global MAX_CTRL_RATE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", help="actuated MJCF (default: taken from hand_config.json)")
    ap.add_argument("--config", help="hand_config.json (default: next to the model)")
    ap.add_argument("--sweep", action="store_true", help="move every joint one at a time")
    ap.add_argument("--demo", action="store_true", help="cycle grasp presets")
    ap.add_argument("--only", action="append", default=[], metavar="NAME",
                    help="with --pose: bend only joints whose name contains NAME (others stay at 0). Repeatable.")
    ap.add_argument("--pose", type=float, metavar="DEG",
                    help="just HOLD every finger/thumb joint at DEG (joint angle: 0 = straight / in the palm plane, "
                         "+ = toward closed); wrist and sideways axes centred. For inspecting the model.")
    ap.add_argument("--headless", type=float, metavar="SECONDS", help="run --sweep/--demo without a window")
    ap.add_argument("--max-rate", type=float, metavar="DEG_PER_S",
                    help=f"fastest any joint target may move (default {math.degrees(MAX_CTRL_RATE):.0f} deg/s)")
    args = ap.parse_args()
    if args.max_rate:
        MAX_CTRL_RATE = math.radians(args.max_rate)

    if args.config:
        cfg_path = args.config
    elif args.model:
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(args.model)), "hand_config.json")
    else:
        cfg_path = "hand_config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    model_path = args.model or os.path.join(os.path.dirname(os.path.abspath(cfg_path)), cfg["model"])

    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    hand = Hand(m, cfg)
    for n in hand.names:                         # start in the open pose, actuators already there
        d.qpos[hand.qadr[n]] = hand.target[n]
        d.ctrl[hand.act[n]] = hand.target[n]
    mujoco.mj_forward(m, d)

    if args.pose is not None:
        for n in hand.names:
            j = hand.J[n]
            if j["role"] == "flex":
                # joint angle itself (0 = straight / its zero), toward the closed side
                deg = args.pose if (not args.only or any(k.lower() in n.lower() for k in args.only)) else 0.0
                hand.target[n] = hand.clamp(n, math.radians(deg) * (1 if j["closed"] >= 0 else -1))
        for n in hand.names:
            d.qpos[hand.qadr[n]] = hand.target[n]
            d.ctrl[hand.act[n]] = hand.target[n]
        mujoco.mj_forward(m, d)
        mode, logic = "hold", None
        bent = [n for n in hand.names if hand.J[n]["role"] == "flex"
                and (not args.only or any(k.lower() in n.lower() for k in args.only))]
        print(f"[pose] holding {len(bent)} joint(s) at {args.pose:g} deg, everything else at 0: "
              f"{bent if args.only else 'all fingers/thumb'}")
    elif args.sweep:
        mode, logic = "auto", Sweep(hand)
    elif args.demo:
        mode, logic = "auto", Demo(hand)
    else:
        pad = Gamepad()
        if pad.kind:
            mode, logic = "pad", PadLogic(hand)
        else:
            mode, logic = "sliders", None
            print("[info] no gamepad found - use the viewer's Control panel sliders (right side).")

    if args.headless:
        if mode not in ("auto", "hold"):
            sys.exit("--headless needs --sweep, --demo or --pose")
        dt, worst = m.opt.timestep, 0.0
        while d.time < args.headless:
            if logic:
                logic.update(dt)
            hand.apply(d, dt)
            mujoco.mj_step(m, d)
            if not np.all(np.isfinite(d.qpos)):
                sys.exit(f"[headless] simulation went unstable at t={d.time:.3f}s")
            worst = max(worst, max(abs(d.qpos[hand.qadr[n]] - d.ctrl[hand.act[n]]) for n in hand.names))
        print(f"[headless] ran {d.time:.1f}s OK, worst |q - ctrl| = {math.degrees(worst):.2f} deg")
        return

    if mjviewer is None:
        sys.exit("mujoco.viewer unavailable (pip install mujoco glfw)")
    with mjviewer.launch_passive(m, d) as viewer:
        wall0, sim0, last = time.perf_counter(), d.time, time.perf_counter()
        while viewer.is_running():
            now = time.perf_counter()
            dt = min(now - last, 0.05)
            last = now
            if mode == "pad":
                logic.update(pad.read(), dt)
            elif mode == "auto":
                logic.update(dt)
            if mode not in ("sliders",):
                hand.apply(d, dt)

            if d.time < sim0:                                    # viewer reset (Backspace)
                wall0, sim0 = now, d.time
            steps = 0
            while d.time - sim0 < now - wall0 and steps < 100:
                mujoco.mj_step(m, d)
                steps += 1
            if steps == 100:                                     # can't keep real time: resync
                wall0, sim0 = now, d.time
            viewer.sync()
            time.sleep(max(0.0, 1 / 90 - (time.perf_counter() - now)))


if __name__ == "__main__":
    main()
