#!/usr/bin/env python3
"""
hand_controller.py
==================
Drive the actuated hand (from add_actuators.py) with a gamepad in the MuJoCo viewer.

  python hand_controller.py robot_actuated.xml            # gamepad control
  python hand_controller.py robot_actuated.xml --sweep    # move every joint one at a time (check axes/limits)
  python hand_controller.py robot_actuated.xml --demo     # cycle open / fist / pinch / point
  python hand_controller.py robot_actuated.xml --pose 45  # hold every finger/thumb joint at 45 deg
  python hand_controller.py robot_actuated.xml --save-pose grip1     # set sliders, close window -> saved
  python hand_controller.py robot_actuated.xml --load-pose grip1     # hold a saved pose
  python hand_controller.py robot_actuated.xml --list-poses
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


# --------------------------------------------------------------------------- keyboard mode
def load_mobile_model(model_path, palm_body):
    """The same model with a MOVABLE forearm: a free joint on the arm's root body, held by a soft
    weld to an invisible mocap body 'base_target'. The keys move the target and the arm follows with
    finite stiffness, so it still collides with / pushes things physically. The model files on disk
    are not changed (a temporary copy is compiled next to the model so mesh paths keep working)."""
    import xml.etree.ElementTree as ET
    m0 = mujoco.MjModel.from_xml_path(model_path)
    b = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, palm_body)
    if b < 0:
        sys.exit(f"palm body '{palm_body}' not in the model")
    while m0.body_parentid[b] != 0:
        b = m0.body_parentid[b]
    root_name = m0.body(b).name
    d0 = mujoco.MjData(m0)
    mujoco.mj_forward(m0, d0)
    fmt = lambda v: " ".join(f"{float(x):.6g}" for x in v)

    tree = ET.parse(model_path)
    root = tree.getroot()
    wb = root.find("worldbody")
    el = next(e for e in wb.iter("body") if e.get("name") == root_name)
    if not any(c.tag == "freejoint" or (c.tag == "joint" and c.get("type") == "free") for c in el):
        el.insert(0, ET.Element("freejoint", name="base_free"))
    tgt = ET.SubElement(wb, "body", name="base_target", mocap="true",
                        pos=fmt(d0.xpos[b]), quat=fmt(d0.xquat[b]))
    ET.SubElement(tgt, "geom", type="sphere", size="0.006", rgba="0.1 0.9 0.3 0.6",
                  contype="0", conaffinity="0", group="2")
    eq = root.find("equality")
    if eq is None:
        eq = ET.SubElement(root, "equality")
    ET.SubElement(eq, "weld", name="base_weld", body1="base_target", body2=root_name,
                  solref="0.02 1", solimp="0.95 0.99 0.001")
    for kf in root.findall("keyframe"):                  # qpos size changed (free joint)
        root.remove(kf)
    tmp = os.path.join(os.path.dirname(os.path.abspath(model_path)), "__mobile_tmp.xml")
    tree.write(tmp)
    try:
        m = mujoco.MjModel.from_xml_path(tmp)
    finally:
        os.remove(tmp)
    return m, root_name


def _quat_axis_angle(axis, angle):
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.asarray(axis, float), float(angle))
    return q


def _rot(q, v):
    out = np.zeros(3)
    mujoco.mju_rotVecQuat(out, np.asarray(v, float), np.asarray(q, float))
    return out


class KeyboardLogic:
    """Moves the arm (via the mocap target) and recalls / saves hand poses. Input is abstract
    (sets of action names) so it can be tested without a window."""
    UP = np.array([0.0, 0.0, 1.0])
    LEASH = 0.05                                   # m
    MOVES = {"fwd": (0, 1, 0), "back": (0, -1, 0), "left": (-1, 0, 0), "right": (1, 0, 0),
             "up": (0, 0, 1), "down": (0, 0, -1)}
    TILTS = {"tilt_f": (0, 1, 0), "tilt_b": (0, -1, 0), "tilt_l": (-1, 0, 0), "tilt_r": (1, 0, 0)}

    def __init__(self, hand, m, d, root_name, poses, poses_path, p_pose=None,
                 move_speed=0.15, turn_speed=60.0, pivot="wrist", mobile=True):
        self.h, self.m, self.d = hand, m, d
        self.mobile = mobile
        self.poses, self.poses_path = poses, poses_path
        self.p_pose = p_pose if p_pose in poses else (list(poses)[-1] if poses else None)
        self.move_speed, self.turn_speed = move_speed, math.radians(turn_speed)
        self.pose_name, self.driving, self.msg = "open", False, ""
        self.root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, root_name)
        palm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, hand.cfg_palm)
        self.pivot_body = m.body_parentid[palm] if pivot == "wrist" and palm >= 0 else self.root
        if mobile:
            tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_target")
            self.mid = m.body_mocapid[tid]
            self.pos0 = d.mocap_pos[self.mid].copy()
            self.quat0 = d.mocap_quat[self.mid].copy()
            qinv = np.zeros(4)
            mujoco.mju_negQuat(qinv, self.quat0)
            self.long_local = _rot(qinv, self.UP)               # the arm's "up" in its own frame

    # ---- hand poses
    def _go(self, name, targets):
        for n, v in targets.items():
            if n in self.h.target:
                self.h.target[n] = self.h.clamp(n, v)
        self.pose_name, self.driving = name, True

    def go_open(self):
        self._go("open", {n: self.h.home(n) for n in self.h.names})

    def go_pose(self, name):
        if name not in self.poses:
            self.msg = f"no pose '{name}' saved"
            return
        base = {n: self.h.home(n) for n in self.h.names}
        base.update(self.poses[name])
        self._go(name, base)

    def save_current(self):
        k = 1
        while f"pose_{k}" in self.poses:
            k += 1
        name = f"pose_{k}"
        self.poses[name] = {n: round(float(self.d.ctrl[self.h.act[n]]), 5) for n in self.h.names}
        with open(self.poses_path, "w") as f:
            json.dump(self.poses, f, indent=2)
        self.msg = f"saved current hand as '{name}' (key {len(self.poses)} if <= 9)"
        print(f"[keyboard] {self.msg} -> {self.poses_path}")

    # ---- per frame
    def apply(self, held, once, dt):
        if "open" in once:
            self.go_open()
        if "pose_p" in once:
            if self.pose_name == self.p_pose and self.p_pose:
                self.go_open()                                   # P again -> back to open
            elif self.p_pose:
                self.go_pose(self.p_pose)
            else:
                self.msg = "no saved pose yet - save one (N key, or --save-pose)"
        for i, name in enumerate(list(self.poses)[:9], start=1):
            if f"pose_{i}" in once:
                self.go_pose(name)
        if "save" in once:
            self.save_current()

        if self.mobile:
            if "reset" in once:
                self.d.mocap_pos[self.mid] = self.pos0
                self.d.mocap_quat[self.mid] = self.quat0
                self.resetting = True
            k = 4.0 if "fast" in held else (0.25 if "fine" in held else 1.0)
            pos = self.d.mocap_pos[self.mid].copy()
            quat = self.d.mocap_quat[self.mid].copy()
            for a, v in self.MOVES.items():
                if a in held:
                    pos += np.array(v, float) * self.move_speed * k * dt
            rots = []
            for a, u in self.TILTS.items():
                if a in held:
                    rots.append(np.cross(self.UP, np.array(u, float)))   # tilt the top toward u
            if "twist_l" in held or "twist_r" in held:
                axis = _rot(quat, self.long_local)
                rots.append(axis if "twist_l" in held else -axis)
            if rots:
                pivot = np.array(self.d.xipos[self.pivot_body]) if self.pivot_body != self.root else pos.copy()
                for axis in rots:
                    dq = _quat_axis_angle(axis, self.turn_speed * k * dt)
                    pos = pivot + _rot(dq, pos - pivot)
                    q2 = np.zeros(4)
                    mujoco.mju_mulQuat(q2, dq, quat)
                    quat = q2 / np.linalg.norm(q2)
            # leash: never let the target run more than LEASH away from the real arm (e.g. when
            # pushing into the floor or a door) - otherwise it winds up and snaps back later
            arm = np.array(self.d.xpos[self.root])
            off = pos - arm
            if getattr(self, "resetting", False):
                if np.linalg.norm(off) < 0.005:
                    self.resetting = False
            elif np.linalg.norm(off) > self.LEASH:
                pos = arm + off * (self.LEASH / np.linalg.norm(off))
            self.d.mocap_pos[self.mid] = pos
            self.d.mocap_quat[self.mid] = quat

        if self.driving and max(abs(self.d.ctrl[self.h.act[n]] - self.h.target[n])
                                for n in self.h.names) < 1e-4:
            self.driving = False                                 # arrived: sliders are yours again

    def status(self):
        lines = []
        if self.mobile:
            dp = (self.d.mocap_pos[self.mid] - self.pos0) * 1000
            up = _rot(self.d.mocap_quat[self.mid], self.long_local)
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(up[2])))))
            lines.append(f"arm offset  x {dp[0]:7.1f}  y {dp[1]:7.1f}  z {dp[2]:7.1f} mm   tilt {tilt:5.1f} deg")
        else:
            lines.append("arm fixed (--fixed-base)")
        names = list(self.poses)[:9]
        lines.append(f"hand: {self.pose_name}{'  (moving)' if self.driving else ''}    "
                     f"P = {self.p_pose or '-'}")
        lines.append("poses: " + ("  ".join(f"{i}={n}" for i, n in enumerate(names, 1)) or "none saved"))
        if self.msg:
            lines.append(self.msg)
        return lines


class KeyWindow:
    """Small pygame window that reads the keyboard (click it to give it focus)."""
    HELP = [
        "CLICK THIS WINDOW, then:",
        "W/S  forward/back     A/D  left/right     +/-  up/down   (PgUp/PgDn too)",
        "Num8/Num2  tilt forward/back    Num4/Num6  tilt left/right   (arrow keys too)",
        "Num7/Num9  twist left/right about the arm   (Q/E too)",
        "Shift = fast   Ctrl = fine     R = reset arm position",
        "P = saved pose (again = open)   0 = open hand   1-9 = saved poses",
        "N = save the current hand as a new pose     Esc = quit",
    ]

    def __init__(self):
        try:
            import pygame
        except ImportError:
            sys.exit("keyboard mode needs pygame:  python -m pip install pygame")
        self.pg = pg = pygame
        pg.init()
        self.screen = pg.display.set_mode((720, 300))
        pg.display.set_caption("Hand keyboard control  (click here first)")
        self.font = pg.font.SysFont("consolas", 16)
        K = lambda n: getattr(pg, n, None)
        hold = {"K_w": "fwd", "K_s": "back", "K_a": "left", "K_d": "right",
                "K_EQUALS": "up", "K_PLUS": "up", "K_KP_PLUS": "up", "K_PAGEUP": "up",
                "K_MINUS": "down", "K_KP_MINUS": "down", "K_PAGEDOWN": "down",
                "K_KP8": "tilt_f", "K_UP": "tilt_f", "K_KP2": "tilt_b", "K_DOWN": "tilt_b",
                "K_KP4": "tilt_l", "K_LEFT": "tilt_l", "K_KP6": "tilt_r", "K_RIGHT": "tilt_r",
                "K_KP7": "twist_l", "K_q": "twist_l", "K_KP9": "twist_r", "K_e": "twist_r",
                "K_LSHIFT": "fast", "K_RSHIFT": "fast", "K_LCTRL": "fine", "K_RCTRL": "fine"}
        once = {"K_p": "pose_p", "K_0": "open", "K_r": "reset", "K_n": "save", "K_ESCAPE": "quit"}
        once.update({f"K_{i}": f"pose_{i}" for i in range(1, 10)})
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
        held = {a for k, a in self.hold.items() if keys[k]}
        return held, once

    def draw(self, status):
        pg = self.pg
        self.screen.fill((24, 24, 28))
        y = 10
        for i, line in enumerate(self.HELP + [""] + status):
            col = (120, 220, 140) if i >= len(self.HELP) else (220, 220, 220)
            self.screen.blit(self.font.render(line, True, col), (12, y))
            y += 22
        pg.display.flip()

    def close(self):
        self.pg.quit()


# --------------------------------------------------------------------------- main
def main():
    global MAX_CTRL_RATE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", help="actuated MJCF (default: taken from hand_config.json)")
    ap.add_argument("--config", help="hand_config.json (default: next to the model)")
    ap.add_argument("--sweep", action="store_true", help="move every joint one at a time")
    ap.add_argument("--demo", action="store_true", help="cycle grasp presets")
    ap.add_argument("--ctrl", metavar="V1,V2,...",
                    help="actuator targets in the viewer's Control-panel order (radians), e.g. copied from the sliders")
    ap.add_argument("--save-pose", metavar="NAME",
                    help="open the viewer with the sliders; when you CLOSE the window the current slider values are "
                         "saved as NAME in poses.json (next to the model). Combine with --ctrl or --load-pose.")
    ap.add_argument("--load-pose", metavar="NAME", help="hold a pose saved with --save-pose")
    ap.add_argument("--list-poses", action="store_true", help="list the saved poses")
    ap.add_argument("--keyboard", action="store_true",
                    help="keyboard mode: move / tilt / twist the whole arm and recall poses (see KEYBOARD above)")
    ap.add_argument("--p-pose", metavar="NAME", help="keyboard: the pose the P key brings up (default: last saved)")
    ap.add_argument("--fixed-base", action="store_true", help="keyboard: keep the arm fixed, only the pose keys")
    ap.add_argument("--move-speed", type=float, default=0.15, metavar="M_PER_S",
                    help="keyboard: arm speed (default 0.15 m/s; Shift x4, Ctrl x0.25)")
    ap.add_argument("--turn-speed", type=float, default=60.0, metavar="DEG_PER_S",
                    help="keyboard: arm turn speed (default 60 deg/s)")
    ap.add_argument("--pivot", choices=["wrist", "base"], default="wrist",
                    help="keyboard: tilt/twist the arm about the wrist ball (default) or its base")
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

    root_name = None
    if args.keyboard and not args.fixed_base:
        m, root_name = load_mobile_model(model_path, cfg["palm_body"])
    else:
        m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    hand = Hand(m, cfg)
    hand.cfg_palm = cfg["palm_body"]
    for n in hand.names:                         # start in the open pose, actuators already there
        d.qpos[hand.qadr[n]] = hand.target[n]
        d.ctrl[hand.act[n]] = hand.target[n]
    mujoco.mj_forward(m, d)

    poses_path = os.path.join(os.path.dirname(os.path.abspath(model_path)), "poses.json")
    poses = json.load(open(poses_path)) if os.path.exists(poses_path) else {}
    if args.list_poses:
        for name, pz in poses.items():
            print(f"{name:<20} " + ", ".join(f"{k.split('__')[-1]}={math.degrees(v):.0f}"
                                             for k, v in pz.items()))
        if not poses:
            print(f"no poses saved yet ({poses_path})")
        return
    by_act = {hand.act[n]: n for n in hand.names}
    preset_pose = None
    if args.load_pose:
        if args.load_pose not in poses:
            sys.exit(f"no pose '{args.load_pose}' in {poses_path}. Saved: {sorted(poses) or 'none'}")
        preset_pose = {n: v for n, v in poses[args.load_pose].items() if n in hand.target}
    if args.ctrl:
        vals = [float(v) for v in args.ctrl.replace(" ", "").split(",") if v]
        if len(vals) != m.nu:
            sys.exit(f"--ctrl has {len(vals)} values but the model has {m.nu} actuators (Control panel order)")
        preset_pose = {by_act[a]: v for a, v in enumerate(vals) if a in by_act}
    if preset_pose:
        for n, v in preset_pose.items():
            hand.target[n] = hand.clamp(n, v)
            d.qpos[hand.qadr[n]] = hand.target[n]
            d.ctrl[hand.act[n]] = hand.target[n]
        mujoco.mj_forward(m, d)

    if args.keyboard:
        if root_name is None:                                     # --fixed-base
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, cfg["palm_body"])
            while m.body_parentid[b] != 0:
                b = m.body_parentid[b]
            root_name = m.body(b).name
        mode = "keyboard"
        logic = KeyboardLogic(hand, m, d, root_name, poses, poses_path, args.p_pose,
                              args.move_speed, args.turn_speed, args.pivot, mobile=not args.fixed_base)
        if args.load_pose:
            logic.go_pose(args.load_pose)
        print("[keyboard] click the small 'Hand keyboard control' window, then use the keys shown in it")
    elif args.save_pose:
        mode, logic = "sliders", None
        print(f"[save-pose] adjust the Control sliders if you like, then CLOSE the window to save '{args.save_pose}'")
    elif preset_pose is not None:
        mode, logic = "hold", None
        print(f"[pose] holding {'pose ' + repr(args.load_pose) if args.load_pose else '--ctrl values'}")
    elif args.pose is not None:
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
    kw = KeyWindow() if mode == "keyboard" else None
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
            elif mode == "keyboard":
                held, once = kw.poll()
                if "quit" in once:
                    break
                with viewer.lock():
                    logic.apply(held, once, dt)
                kw.draw(logic.status())
            if mode == "keyboard":
                if logic.driving:                    # only while going to a pose: sliders stay usable
                    hand.apply(d, dt)
            elif mode not in ("sliders",):
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

    if kw is not None:
        kw.close()

    if args.save_pose:
        poses[args.save_pose] = {by_act[a]: round(float(d.ctrl[a]), 5) for a in range(m.nu) if a in by_act}
        with open(poses_path, "w") as f:
            json.dump(poses, f, indent=2)
        print(f"[save-pose] saved '{args.save_pose}' ({len(poses[args.save_pose])} joints) -> {poses_path}")
        for n, v in poses[args.save_pose].items():
            print(f"   {n.split('__')[-1]:<34} {math.degrees(v):7.1f} deg")


if __name__ == "__main__":
    main()
