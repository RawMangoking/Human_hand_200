#!/usr/bin/env python3
"""
door_env.py
===========
Gymnasium environment: the v5 hand (on a soft-welded, movable arm) in front of the v4 door.

The scene is built at start-up by merging the two MuJoCo models you already have
  hand : Mujoko/Hand_forearm_v5/mjcf/Hand_forearm_actuated.xml  + hand_config.json   (add_actuators.py)
  door : Mujoko/Full_door_v4/mjcf/Full_door_v4_sim.xml         + door_config.json    (door_setup.py)
All names get a prefix (hand/..., door/...), mesh paths are made absolute, the files are not changed.
The merged model is also written to  Mujoko/Hand_Door_env/scene.xml  so you can open it in the viewer.

The arm is carried by a soft weld to an invisible mocap target (like hand_controller.py --keyboard),
so it pushes and collides physically. It starts with the palm facing the door, in front of the handle.

TASKS (one small action set each; every action is in [-1, 1])
  reach   move the palm to the handle                 arm_pos(3) arm_rot(3)                              = 6
  handle  grasp the handle and turn it open           arm_pos(3) arm_rot(3) wrist(2) index mrp thumb opp = 12
  door    handle already turned: push/pull door open  arm_pos(3) grip                                    = 4
DOMAIN RANDOMIZATION: DoorEnv(randomize_physics=True) samples door mass, hinge damping / friction, handle
friction and grip friction each reset (PHYSICS_RANGES, multipliers of the nominal door); DoorEnv(physics={...})
fixes them (e.g. held-out test doors). info["physics"] reports what was used.

COLLISIONS (default): hand parts may overlap each other; the hand collides with the door, handle, frame and
floor. --self-collision makes the hand collide with itself too; --no-hand-door-collision lets it pass through.

  arm_pos : target moves up to 1 cm per step        arm_rot : up to 3 deg per step (world x, y, z)
  wrist   : flexion, tilt          index / mrp / thumb / opp / grip : finger curl
            all of these are RATES: 0 = hold, +-1 = move 10 % of the range per step
  mrp = middle + ring + pinky together (coupled in the action mapping); opp = thumb base (CMC)
  grip = index + mrp + thumb together

Usage
  python door_env.py --task reach --view          # just look at the start pose (no policy)
  python door_env.py --task reach                 # random actions in the viewer (checks the scene)
  python door_env.py --task handle --check        # gymnasium env checker, no window
  python door_env.py --task door --scripted       # a hand-written policy, shows the task is solvable

  from door_env import DoorEnv
  env = DoorEnv(task="reach")
  obs, info = env.reset(seed=0)
  obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""
import argparse
import copy
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    sys.exit("needs gymnasium:  python -m pip install gymnasium")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_HAND = os.path.join(REPO, "Mujoko", "Hand_forearm_v5", "mjcf", "Hand_forearm_actuated.xml")
DEFAULT_DOOR = os.path.join(REPO, "Mujoko", "Full_door_v4", "mjcf", "Full_door_v4_sim.xml")

REF_ATTRS = {"name", "class", "childclass", "body", "body1", "body2", "joint", "joint1", "joint2", "geom",
             "geom1", "geom2", "site", "site1", "site2", "mesh", "material", "texture", "tendon", "actuator",
             "objname", "refname", "target", "hfield", "skin", "cranksite", "slidersite", "jointinparent"}
FILE_TAGS = {"mesh": "meshdir", "texture": "texturedir", "hfield": "assetdir", "skin": "assetdir"}


# --------------------------------------------------------------------------- scene building
def _prefixed_tree(path, prefix):
    """Parse an MJCF, prefix every name / name reference, make asset paths absolute."""
    tree = ET.parse(path)
    root = tree.getroot()
    base = os.path.dirname(os.path.abspath(path))
    comp = root.find("compiler")
    dirs = {"meshdir": "", "texturedir": "", "assetdir": ""}
    if comp is not None:
        for k in dirs:
            dirs[k] = comp.get(k, "")
        if comp.get("assetdir") and not comp.get("meshdir"):
            dirs["meshdir"] = comp.get("assetdir")
        if comp.get("assetdir") and not comp.get("texturedir"):
            dirs["texturedir"] = comp.get("assetdir")
    for el in root.iter():
        if el.tag in FILE_TAGS and el.get("file"):
            el.set("file", os.path.normpath(os.path.join(base, dirs[FILE_TAGS[el.tag]], el.get("file"))))
        for a in list(el.attrib):
            if a in REF_ATTRS and el.get(a):
                if a == "class" and el.tag == "default" and el.get(a) == "main":
                    continue
                el.set(a, prefix + el.get(a))
    # unnamed top-level <default> -> a named class, applied to this model's bodies via childclass
    top_default = root.find("default")
    cls = None
    if top_default is not None and top_default.get("class") is None:
        cls = prefix + "main"
        top_default.set("class", cls)
        wb = root.find("worldbody")
        if wb is not None:
            for b in wb.findall("body"):
                if not b.get("childclass"):
                    b.set("childclass", cls)
            for g in wb.findall("geom") + wb.findall("site"):
                if not g.get("class"):
                    g.set("class", cls)
    return root, cls


def _quat_from_mat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).flatten())
    return q


def _fmt(v):
    return " ".join(f"{float(x):.6g}" for x in v)


def hand_geometry(hand_path, cfg):
    """Palm centre, palm normal (palm side), hand length direction and root pose, in the hand model."""
    m = mujoco.MjModel.from_xml_path(hand_path)
    d = mujoco.MjData(m)
    for n, j in cfg["joints"].items():
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        if jid >= 0:
            d.qpos[m.jnt_qposadr[jid]] = j["open"] if j["role"] == "flex" else j["neutral"]
    mujoco.mj_forward(m, d)
    palm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, cfg["palm_body"])
    root = palm
    while m.body_parentid[root] != 0:
        root = m.body_parentid[root]
    mcps = [names[0] for g, names in cfg["fingers"].items() if not g.startswith("thumb") and names]
    K = np.sum([d.xaxis[m.joint(n).id] for n in mcps], axis=0)
    A = np.mean([d.xanchor[m.joint(n).id] for n in mcps], axis=0)
    wr = cfg.get("wrist_axes", {}).get("flex")
    base_pt = np.array(d.xanchor[m.joint(wr).id]) if wr else np.array(d.xpos[palm])
    L = A - base_pt
    L /= np.linalg.norm(L)
    K = K - L * (K @ L)
    K /= np.linalg.norm(K)
    P = np.cross(K, L)                      # + flexion moves fingertips toward the palm side
    P /= np.linalg.norm(P)
    return dict(palm_c=np.array(d.xipos[palm]), P=P, L=L, root=m.body(root).name,
                root_pos=np.array(d.xpos[root]), root_quat=np.array(d.xquat[root]))


def door_geometry(door_path, cfg):
    m = mujoco.MjModel.from_xml_path(door_path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    jh, jd = m.joint(cfg["hinge"]).id, m.joint(cfg["handle"]).id
    door_b, hand_b = m.jnt_bodyid[jh], m.jnt_bodyid[jd]
    p = np.array(d.xanchor[jh])
    z = np.array([0.0, 0.0, 1.0])
    u = np.array(d.xipos[door_b]) - p
    u -= z * (u @ z)
    u /= np.linalg.norm(u)
    v = np.cross(z, u)
    grasp = np.array(d.xipos[hand_b])                      # handle centre of mass = where to grip
    side = 1.0 if (grasp - p) @ v >= 0 else -1.0            # the door face the handle is on
    # which hinge direction moves the door AWAY from the handle side (= push)?
    adr = m.jnt_qposadr[jh]
    c0 = np.array(d.xipos[door_b])
    d.qpos[adr] += 0.05
    mujoco.mj_forward(m, d)
    push_sign = -1.0 if (np.array(d.xipos[door_b]) - c0) @ v * side > 0 else 1.0
    return dict(grasp=grasp, normal=v * side, push_sign=push_sign)


def set_collisions(model, hand_self=False, hand_door=True):
    """Collision rules with MuJoCo bitmasks (a pair collides if contype1 & conaffinity2 or contype2 & conaffinity1):
       hand geoms: contype 2, conaffinity 1  -> hand-hand: 2&1 = 0 -> no self-collision
       other geoms: conaffinity |= 2         -> hand-door / hand-floor: 2&(..|2) -> collide
       Geoms that never collided (visual meshes) are left alone."""
    hand_b = {b for b in range(model.nbody) if model.body(b).name.startswith("hand/")}
    n_hand = n_other = 0
    for g in range(model.ngeom):
        if not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        if model.geom_bodyid[g] in hand_b:
            if not hand_door:
                model.geom_contype[g], model.geom_conaffinity[g] = (1, 1) if hand_self else (0, 0)
            elif hand_self:
                model.geom_contype[g], model.geom_conaffinity[g] = 3, 3
            else:
                model.geom_contype[g], model.geom_conaffinity[g] = 2, 1
            n_hand += 1
        else:
            if hand_door:
                model.geom_conaffinity[g] = int(model.geom_conaffinity[g]) | 2
            n_other += 1
    return n_hand, n_other


def build_scene(hand_path, door_path, standoff=0.30):
    hand_cfg_path = os.path.join(os.path.dirname(hand_path), "hand_config.json")
    door_cfg_path = os.path.join(os.path.dirname(door_path), "door_config.json")
    hcfg, dcfg = json.load(open(hand_cfg_path)), json.load(open(door_cfg_path))
    hg, dg = hand_geometry(hand_path, hcfg), door_geometry(door_path, dcfg)

    # initial arm placement: palm `standoff` in front of the handle, palm facing the door (yaw about z)
    target_palm = dg["grasp"] + dg["normal"] * standoff
    want = -dg["normal"]
    have = hg["P"].copy()
    have[2] = 0.0
    want = want.copy()
    want[2] = 0.0
    yaw = math.atan2(have[0] * want[1] - have[1] * want[0], have[0] * want[0] + have[1] * want[1])
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    r = hg["palm_c"] - hg["root_pos"]
    root_pos = target_palm - Rz @ r
    Rroot = np.zeros(9)
    mujoco.mju_quat2Mat(Rroot, hg["root_quat"])
    root_quat = _quat_from_mat(Rz @ Rroot.reshape(3, 3))

    droot, _ = _prefixed_tree(door_path, "door/")
    hroot, _ = _prefixed_tree(hand_path, "hand/")
    for r_ in (droot, hroot):
        c = r_.find("compiler")
        if c is not None:
            for k in ("meshdir", "texturedir", "assetdir"):
                c.attrib.pop(k, None)
    a_d = (droot.find("compiler").get("angle", "degree") if droot.find("compiler") is not None else "degree")
    a_h = (hroot.find("compiler").get("angle", "degree") if hroot.find("compiler") is not None else "degree")
    if a_d != a_h:
        sys.exit(f"hand and door use different angle units ({a_h} vs {a_d}) - rebuild one of them")

    scene = ET.Element("mujoco", model="hand_door")
    comp = copy.deepcopy(droot.find("compiler")) if droot.find("compiler") is not None else ET.Element("compiler")
    scene.append(comp)
    ET.SubElement(scene, "option", timestep="0.002", integrator="implicitfast")
    for tag in ("default", "asset"):
        merged = ET.SubElement(scene, tag)
        for r_ in (droot, hroot):
            for el in r_.findall(tag):
                if tag == "default":
                    merged.append(el)
                else:
                    for c in el:
                        merged.append(c)
        if not len(merged):
            scene.remove(merged)
    wb = ET.SubElement(scene, "worldbody")
    ET.SubElement(wb, "light", name="scene_light", pos="0 0 4", dir="0 0 -1", directional="true")
    for c in droot.find("worldbody"):
        wb.append(c)
    hroot_body = None
    for c in hroot.find("worldbody"):
        if c.tag == "body" and c.get("name") == "hand/" + hg["root"]:
            hroot_body = c
        wb.append(c)
    if hroot_body is None:
        sys.exit("could not find the hand's root body")
    for a in ("quat", "euler", "axisangle", "xyaxes", "zaxis"):
        hroot_body.attrib.pop(a, None)
    hroot_body.set("pos", _fmt(root_pos))
    hroot_body.set("quat", _fmt(root_quat))
    if not any(ch.tag == "freejoint" for ch in hroot_body):
        hroot_body.insert(0, ET.Element("freejoint", name="hand/base_free"))
    for b in hroot_body.iter("body"):                   # the arm carries its own weight, like a real robot arm
        b.set("gravcomp", "1")
    tgt = ET.SubElement(wb, "body", name="hand/base_target", mocap="true", pos=_fmt(root_pos), quat=_fmt(root_quat))
    ET.SubElement(tgt, "geom", type="sphere", size="0.006", rgba="0.1 0.9 0.3 0.5", contype="0",
                  conaffinity="0", group="2")
    for tag in ("contact", "equality", "tendon", "actuator", "sensor"):
        merged = ET.SubElement(scene, tag)
        for r_ in (droot, hroot):
            for el in r_.findall(tag):
                for c in el:
                    merged.append(c)
        if tag == "equality":
            ET.SubElement(merged, "weld", name="hand/base_weld", body1="hand/base_target",
                          body2="hand/" + hg["root"], solref="0.05 1", solimp="0.95 0.99 0.001")
        if not len(merged):
            scene.remove(merged)
    xml = ET.tostring(scene, encoding="unicode")
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(door_path)))),
                           "Hand_Door_env")
    try:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "scene.xml"), "w") as f:
            f.write(xml)
    except OSError:
        pass
    model = mujoco.MjModel.from_xml_string(xml)
    return model, hcfg, dcfg, dict(hand=hg, door=dg, root_pos=root_pos, root_quat=root_quat)


# --------------------------------------------------------------------------- environment
class DoorEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}
    TASKS = {
        "reach": ["arm_pos", "arm_rot"],
        "handle": ["arm_pos", "arm_rot", "wrist", "index", "mrp", "thumb", "opp"],
        "door": ["arm_pos", "grip"],
    }
    SIZES = {"arm_pos": 3, "arm_rot": 3, "wrist": 2, "index": 1, "mrp": 1, "thumb": 1, "opp": 1, "grip": 1}
    LEASH = 0.02             # m: how far the arm target may run ahead of the arm
    STEP_FRAC = 0.10         # finger / wrist actions: fraction of the range per step at action = 1
    # domain randomization: each value is a multiplier of the door's nominal value (from door_setup.py)
    PHYSICS_RANGES = {"door_mass": (0.6, 1.4), "hinge_damping": (0.5, 2.0), "hinge_friction": (0.5, 2.0),
                      "handle_friction": (0.5, 2.0), "grip_friction": (0.6, 1.4)}

    def __init__(self, task="reach", hand_path=DEFAULT_HAND, door_path=DEFAULT_DOOR, render_mode=None,
                 control_hz=25, max_steps=250, door_dir="push", randomize=True, latch=True, unlock_frac=0.8,
                 hand_self_collision=False, hand_door_collision=True,
                 randomize_physics=False, physics=None, physics_ranges=None):
        assert task in self.TASKS, f"task must be one of {list(self.TASKS)}"
        self.task, self.render_mode, self.randomize = task, render_mode, randomize
        self.model, self.hcfg, self.dcfg, self.geo = build_scene(hand_path, door_path)
        # collision rules: hand may overlap itself, hand collides with the door / handle / frame / floor
        n_hand, n_door = set_collisions(self.model, hand_self=hand_self_collision, hand_door=hand_door_collision)
        if n_hand == 0 and hand_door_collision:
            # the hand was exported without collision shapes: use its visible meshes for collision
            hb = {b for b in range(self.model.nbody) if self.model.body(b).name.startswith("hand/")}
            for g in range(self.model.ngeom):
                if self.model.geom_bodyid[g] in hb:
                    self.model.geom_contype[g], self.model.geom_conaffinity[g] = 2, 1
            n_hand = sum(1 for g in range(self.model.ngeom) if self.model.geom_bodyid[g] in hb)
            print(f"[door_env] the hand had NO collision shapes - using its {n_hand} visible meshes for collision")
        self.n_collision_geoms = (n_hand, n_door)
        # stiffer hand contacts: the hand presses on the door instead of sinking into it
        for g in range(self.model.ngeom):
            if self.model.body(self.model.geom_bodyid[g]).name.startswith("hand/") and \
                    (self.model.geom_contype[g] or self.model.geom_conaffinity[g]):
                self.model.geom_solref[g] = [0.005, 1.0]
                self.model.geom_solimp[g] = [0.95, 0.99, 0.001, 0.5, 2.0]
                self.model.geom_solmix[g] = 10.0
        m = self.model
        self.data = mujoco.MjData(m)
        self.frame_skip = max(1, int(round(1.0 / (control_hz * m.opt.timestep))))
        self.max_steps = max_steps
        J = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        B = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        A = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)

        # hand joints / actuators (prefixed names)
        self.hj = {}
        for n, j in self.hcfg["joints"].items():
            jid, aid = J("hand/" + n), A("hand/" + j["actuator"])
            if jid < 0 or aid < 0:
                sys.exit(f"hand joint/actuator '{n}' missing in the merged scene")
            self.hj[n] = dict(cfg=j, qadr=m.jnt_qposadr[jid], dof=m.jnt_dofadr[jid], act=aid)
        f = self.hcfg["fingers"]
        thumb = f.get("thumb", [])
        self.groups = {"index": f.get("index", []),
                       "mrp": f.get("middle", []) + f.get("ring", []) + f.get("pinky", []),
                       "thumb": thumb[1:], "opp": thumb[:1]}
        wa = self.hcfg.get("wrist_axes", {})
        self.wrist = [wa[k] for k in ("flex", "deviation") if k in wa]
        self.palm = B("hand/" + self.hcfg["palm_body"])
        self.tips = [B(m.body(m.jnt_bodyid[J("hand/" + names[-1])]).name) for g, names in f.items() if names]
        self.hand_bodies = {b for b in range(m.nbody) if m.body(b).name.startswith("hand/")}

        # door
        self.jh, self.jd = J("door/" + self.dcfg["hinge"]), J("door/" + self.dcfg["handle"])
        self.qh, self.qd = m.jnt_qposadr[self.jh], m.jnt_qposadr[self.jd]
        self.h0, self.d0 = float(m.qpos0[self.qh]), float(m.qpos0[self.qd])
        self.handle_body = m.jnt_bodyid[self.jd]
        lo, hi = m.jnt_range[self.jd]
        self.handle_open = hi if (hi - self.d0) >= (self.d0 - lo) else lo     # its opening side
        lo, hi = m.jnt_range[self.jh]
        sign = self.geo["door"]["push_sign"] * (1 if door_dir == "push" else -1)
        self.door_goal_sign = sign
        self.door_goal = self.h0 + sign * math.radians(60)
        self.latch_eq = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "door/door_latch")
        # env latch: the door is held closed (both ways) until the handle is turned unlock_frac of its travel
        self.latch, self.unlock_frac = latch, unlock_frac
        self.hinge_range = m.jnt_range[self.jh].copy()
        self.hinge_limited = bool(m.jnt_limited[self.jh])

        # arm (mocap target)
        self.mid = m.body_mocapid[B("hand/base_target")]
        self.free_q = m.jnt_qposadr[J("hand/base_free")]
        self.free_v = m.jnt_dofadr[J("hand/base_free")]

        n_act = sum(self.SIZES[k] for k in self.TASKS[task])
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_act,), dtype=np.float32)
        self._last_action = np.zeros(n_act, dtype=np.float32)
        obs = self._obs()
        self.observation_space = spaces.Box(-1e3, 1e3, shape=obs.shape, dtype=np.float32)
        self._viewer, self._renderer = None, None
        self.steps = 0
        self.hand_door_collision = hand_door_collision
        self.action_layout = [(k, self.SIZES[k]) for k in self.TASKS[task]]

        # ---- door physics: nominal values, randomization ranges (multipliers of the nominal value)
        self.door_body = m.jnt_bodyid[self.jh]
        self.dof_h, self.dof_d = m.jnt_dofadr[self.jh], m.jnt_dofadr[self.jd]
        self.handle_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == self.handle_body]
        self._nominal = dict(door_mass=float(m.body_mass[self.door_body]),
                             door_inertia=m.body_inertia[self.door_body].copy(),
                             hinge_damping=float(m.dof_damping[self.dof_h]),
                             hinge_friction=float(m.dof_frictionloss[self.dof_h]),
                             handle_friction=float(m.dof_frictionloss[self.dof_d]),
                             grip_friction=m.geom_friction[self.handle_geoms, 0].copy())
        self.physics_ranges = dict(self.PHYSICS_RANGES, **(physics_ranges or {}))
        self.randomize_physics, self.fixed_physics = randomize_physics, physics
        self.physics = {k: 1.0 for k in self.physics_ranges}

    # ---- helpers
    def _home(self, n):
        c = self.hj[n]["cfg"]
        return c["open"] if c["role"] == "flex" else c["neutral"]

    def _set_curl(self, group, c):
        for n in self.groups[group]:
            j = self.hj[n]["cfg"]
            self._target[n] = j["open"] + (j["closed"] - j["open"]) * float(np.clip(c, 0, 1))

    def _set_bipolar(self, n, x):
        j = self.hj[n]["cfg"]
        lo, hi = j["range"]
        nu = j["neutral"]
        self._target[n] = nu + x * (hi - nu) if x >= 0 else nu + x * (nu - lo)

    def _place_arm(self, palm_offset, yaw):
        """Put the mocap target AND the free joint so the palm is `palm_offset` from the handle."""
        g = self.geo
        Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        R0 = np.zeros(9)
        mujoco.mju_quat2Mat(R0, g["root_quat"])
        R = Rz @ R0.reshape(3, 3)
        palm0 = g["door"]["grasp"] + g["door"]["normal"] * 0.30        # where build_scene put the palm
        pos = g["door"]["grasp"] + palm_offset + Rz @ (g["root_pos"] - palm0)
        quat = _quat_from_mat(R)
        d = self.data
        d.mocap_pos[self.mid], d.mocap_quat[self.mid] = pos, quat
        d.qpos[self.free_q:self.free_q + 3], d.qpos[self.free_q + 3:self.free_q + 7] = pos, quat

    def _contacts(self):
        """(fingertips touching the handle, palm touching the handle, any hand part touching the handle)"""
        m, d = self.model, self.data
        tips, palm, any_ = set(), False, False
        for c in d.contact[:d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if self.handle_body in (b1, b2):
                other = b2 if b1 == self.handle_body else b1
                if other in self.hand_bodies:
                    any_ = True
                    if other in self.tips:
                        tips.add(other)
                    if other == self.palm:
                        palm = True
        return tips, palm, any_

    def _obs(self):
        m, d = self.model, self.data
        palm_c = np.array(d.xipos[self.palm])
        grasp = np.array(d.xipos[self.handle_body])
        R = np.array(d.xmat[self.palm]).reshape(3, 3)
        q = []
        for n, h in self.hj.items():
            lo, hi = h["cfg"]["range"]
            q.append(2 * (d.qpos[h["qadr"]] - lo) / max(hi - lo, 1e-6) - 1)
        tips, palm_t, _ = self._contacts() if d.ncon else (set(), False, False)
        tip_flags = [1.0 if b in tips else 0.0 for b in self.tips]
        door = [d.qpos[self.qh] - self.h0, d.qpos[self.qd] - self.d0, d.qvel[m.jnt_dofadr[self.jh]],
                d.qvel[m.jnt_dofadr[self.jd]]]
        off = d.mocap_pos[self.mid] - d.xpos[m.body_weldid[self.palm]] if False else \
            d.mocap_pos[self.mid] - d.qpos[self.free_q:self.free_q + 3]
        o = np.concatenate([grasp - palm_c, R[:, 0], R[:, 2], q, door, tip_flags, [1.0 if palm_t else 0.0],
                            off, self._last_action])
        return np.clip(o, -1e3, 1e3).astype(np.float32)

    # ---- door physics (domain randomization)
    def _apply_physics(self, mult):
        m, n = self.model, self._nominal
        m.body_mass[self.door_body] = n["door_mass"] * mult["door_mass"]
        m.body_inertia[self.door_body] = n["door_inertia"] * mult["door_mass"]
        m.dof_damping[self.dof_h] = n["hinge_damping"] * mult["hinge_damping"]
        m.dof_frictionloss[self.dof_h] = n["hinge_friction"] * mult["hinge_friction"]
        m.dof_frictionloss[self.dof_d] = n["handle_friction"] * mult["handle_friction"]
        m.geom_friction[self.handle_geoms, 0] = n["grip_friction"] * mult["grip_friction"]
        self.physics = dict(mult)

    def _sample_physics(self):
        if self.fixed_physics is not None:
            mult = {k: 1.0 for k in self.physics_ranges}
            mult.update(self.fixed_physics)
        elif self.randomize_physics:
            mult = {k: float(self.np_random.uniform(lo, hi)) for k, (lo, hi) in self.physics_ranges.items()}
        else:
            mult = {k: 1.0 for k in self.physics_ranges}
        self._apply_physics(mult)

    # ---- start pose: never start with the hand inside the door
    def hand_overlaps(self, depth=1e-4):
        """[(hand body, other body, depth m)] for hand parts overlapping anything that is not the hand."""
        m, d = self.model, self.data
        out = []
        for c in d.contact[:d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if (b1 in self.hand_bodies) != (b2 in self.hand_bodies) and c.dist < -depth:
                hb, ob = (b1, b2) if b1 in self.hand_bodies else (b2, b1)
                out.append((m.body(hb).name, m.body(ob).name, float(-c.dist)))
        return out

    def _place_clear(self, normal_offset, lateral, yaw):
        """Place the palm `normal_offset` in front of the handle (+ lateral offset); if any hand part overlaps
        the door/frame/handle/floor, move the arm back along the door normal 1 cm at a time until it is clear."""
        nrm = self.geo["door"]["normal"]
        pushed = 0.0
        for _ in range(120):
            self._place_arm(nrm * (normal_offset + pushed) + lateral, yaw)
            mujoco.mj_forward(self.model, self.data)
            if not self.hand_door_collision or not self.hand_overlaps():
                break
            pushed += 0.01
        self.start_pushed_back = pushed
        return pushed

    # ---- gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        self._target = {n: self._home(n) for n in self.hj}
        self._curl = {g: 0.0 for g in ("index", "mrp", "thumb", "opp")}
        self._wpos = [0.0 for _ in self.wrist]
        for n, h in self.hj.items():
            d.qpos[h["qadr"]] = self._target[n]
            d.ctrl[h["act"]] = self._target[n]
        rnd = self.np_random
        self._sample_physics()
        noise = (rnd.uniform(-1, 1, 3) * 0.05) if self.randomize else np.zeros(3)
        yaw = rnd.uniform(-1, 1) * math.radians(10) if self.randomize else 0.0
        nrm = self.geo["door"]["normal"]
        lateral = noise - nrm * (noise @ nrm)                   # sideways / vertical part of the noise
        along = float(noise @ nrm)
        if self.task == "door":                                 # door: handle already turned open
            d.qpos[self.qd] = self.handle_open
            if self.latch_eq >= 0:
                d.eq_active[self.latch_eq] = 0
        start = {"reach": (0.30 + along, lateral, yaw),
                 "handle": (0.07 + 0.3 * along, 0.3 * lateral, 0.3 * yaw),
                 "door": (0.05 + 0.3 * along, 0.3 * lateral, 0.3 * yaw)}[self.task]
        self._place_clear(*start)
        self._update_latch()
        mujoco.mj_forward(m, d)
        self.steps = 0
        self._last_action[:] = 0
        self._prev = self._progress()
        return self._obs(), self._info()

    def _apply_action(self, a):
        d = self.data
        k = 0
        for name in self.TASKS[self.task]:
            n = self.SIZES[name]
            x = a[k:k + n]
            k += n
            if name == "arm_pos":
                d.mocap_pos[self.mid] = d.mocap_pos[self.mid] + 0.01 * x
            elif name == "arm_rot":
                for axis, val in zip(np.eye(3), x):
                    if abs(val) > 1e-6:
                        dq = np.zeros(4)
                        mujoco.mju_axisAngle2Quat(dq, axis, math.radians(3.0) * float(val))
                        qn = np.zeros(4)
                        mujoco.mju_mulQuat(qn, dq, d.mocap_quat[self.mid])
                        d.mocap_quat[self.mid] = qn / np.linalg.norm(qn)
            elif name == "wrist":
                for i, (wn, val) in enumerate(zip(self.wrist, x)):
                    self._wpos[i] = float(np.clip(self._wpos[i] + self.STEP_FRAC * float(val), -1, 1))
                    self._set_bipolar(wn, self._wpos[i])
            elif name in ("index", "mrp", "thumb", "opp"):
                self._curl[name] = float(np.clip(self._curl[name] + self.STEP_FRAC * float(x[0]), 0, 1))
                self._set_curl(name, self._curl[name])
            elif name == "grip":
                for g in ("index", "mrp", "thumb"):
                    self._curl[g] = float(np.clip(self._curl[g] + self.STEP_FRAC * float(x[0]), 0, 1))
                    self._set_curl(g, self._curl[g])
        # leash: the target never runs more than LEASH ahead of the real arm. With the soft weld this caps the
        # arm's push at ~20 N, so it cannot force the hand through the door (a real arm is force-limited too)
        arm = d.qpos[self.free_q:self.free_q + 3]
        off = d.mocap_pos[self.mid] - arm
        n = np.linalg.norm(off)
        if n > self.LEASH:
            d.mocap_pos[self.mid] = arm + off * self.LEASH / n

    def _update_latch(self):
        m, d = self.model, self.data
        if not self.latch:
            return
        frac = (d.qpos[self.qd] - self.d0) / (self.handle_open - self.d0 + 1e-9)
        closed = abs(d.qpos[self.qh] - self.h0) < math.radians(1.5)
        if frac < self.unlock_frac and closed:
            m.jnt_range[self.jh] = [self.h0 - 1e-3, self.h0 + 1e-3]         # latched
            m.jnt_limited[self.jh] = 1
        else:
            m.jnt_range[self.jh] = self.hinge_range
            m.jnt_limited[self.jh] = self.hinge_limited

    def _progress(self):
        d = self.data
        palm_c = np.array(d.xipos[self.palm])
        dist = float(np.linalg.norm(np.array(d.xipos[self.handle_body]) - palm_c))
        R = np.array(d.xmat[self.palm]).reshape(3, 3)
        return dict(dist=dist,
                    handle=float((d.qpos[self.qd] - self.d0) / (self.handle_open - self.d0 + 1e-9)),
                    door=float((d.qpos[self.qh] - self.h0) * self.door_goal_sign))

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
        self._apply_action(a)
        m, d = self.model, self.data
        rate = 4.0 * self.frame_skip * m.opt.timestep            # rad per control step
        for n, h in self.hj.items():
            d.ctrl[h["act"]] += float(np.clip(self._target[n] - d.ctrl[h["act"]], -rate, rate))
        for _ in range(self.frame_skip):
            self._update_latch()
            mujoco.mj_step(m, d)
        self.steps += 1
        p = self._progress()
        tips, palm_t, touch = self._contacts()
        # ---- rewards
        if self.task == "reach":
            r = -p["dist"] + 5.0 * (self._prev["dist"] - p["dist"])
            success = p["dist"] < 0.04
        elif self.task == "handle":
            r = 10.0 * (p["handle"] - self._prev["handle"]) - 0.5 * p["dist"] + 0.05 * len(tips) + 0.05 * palm_t
            success = p["handle"] > 0.8
        else:
            r = 5.0 * (p["door"] - self._prev["door"]) + 0.02 * touch
            success = p["door"] > math.radians(60)
        r -= 0.01 * float(np.sum(a ** 2))
        if success:
            r += 10.0
        self._prev = p
        self._last_action = a
        terminated = bool(success)
        truncated = self.steps >= self.max_steps
        if not np.all(np.isfinite(d.qpos)):
            terminated, r = True, -10.0
        if self.render_mode == "human":
            self.render()
        return self._obs(), float(r), terminated, truncated, self._info(p, success, touch)

    def _info(self, p=None, success=False, touch=False):
        p = p or self._progress()
        return dict(palm_to_handle=p["dist"], handle_frac=p["handle"],
                    door_deg=math.degrees(p["door"]), success=bool(success), touching_handle=bool(touch),
                    physics=dict(self.physics), start_pushed_back=getattr(self, "start_pushed_back", 0.0))

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer as mv
                self._viewer = mv.launch_passive(self.model, self.data)
                self._viewer.cam.lookat[:] = self.geo["door"]["grasp"]
                self._viewer.cam.distance = 1.6
                self._viewer.cam.azimuth, self._viewer.cam.elevation = 135, -15
            self._viewer.sync()
        elif self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, 480, 640)
                self._cam = mujoco.MjvCamera()
                self._cam.lookat[:] = self.geo["door"]["grasp"]
                self._cam.distance, self._cam.azimuth, self._cam.elevation = 1.6, 135, -15
            self._renderer.update_scene(self.data, self._cam)
            return self._renderer.render()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# --------------------------------------------------------------------------- scripted demo policies
def scripted_action(env):
    """Simple hand-written policies (shows each task is solvable; also a baseline)."""
    d = env.data
    palm_c = np.array(d.xipos[env.palm])
    grasp = np.array(d.xipos[env.handle_body])
    a = np.zeros(env.action_space.shape, dtype=np.float32)
    to = np.clip((grasp - palm_c) / 0.01, -1, 1)
    if env.task == "reach":
        a[:3] = to
    elif env.task == "handle":
        near = np.linalg.norm(grasp - palm_c) < 0.05
        a[8:12] = 1.0 if near else -1.0                                           # grip when close
        if not near:
            a[:3] = to
        else:                                                                     # move along the lever's arc
            ax, piv = np.array(d.xaxis[env.jd]), np.array(d.xanchor[env.jd])
            tang = np.cross(ax, grasp - piv) * np.sign(env.handle_open - env.d0)
            n = np.linalg.norm(tang)
            a[:3] = np.clip(tang / (n + 1e-9) + 0.5 * to, -1, 1) if n > 1e-6 else to
    else:
        a[:3] = -env.geo["door"]["normal"] * env.door_goal_sign * env.geo["door"]["push_sign"]
        a[3] = -1.0
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=list(DoorEnv.TASKS), default="reach")
    ap.add_argument("--hand", default=DEFAULT_HAND)
    ap.add_argument("--door", default=DEFAULT_DOOR)
    ap.add_argument("--check", action="store_true", help="run gymnasium's env checker + a short random rollout")
    ap.add_argument("--scripted", action="store_true", help="use the hand-written policy instead of random")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--view", action="store_true", help="just open the environment at its start pose (no policy)")
    ap.add_argument("--randomize-physics", action="store_true", help="domain randomization of the door physics")
    ap.add_argument("--self-collision", action="store_true", help="hand parts also collide with each other")
    ap.add_argument("--no-hand-door-collision", action="store_true", help="hand passes through the door")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    if args.check:
        from gymnasium.utils.env_checker import check_env
        env = DoorEnv(task=args.task, hand_path=args.hand, door_path=args.door,
                      hand_self_collision=args.self_collision, hand_door_collision=not args.no_hand_door_collision)
        check_env(env, skip_render_check=True)
        print(f"[check] OK  task={args.task}  obs {env.observation_space.shape}  action {env.action_space.shape}  "
              f"control {1 / (env.frame_skip * env.model.opt.timestep):.0f} Hz")
        return
    if args.view:
        import time
        import mujoco.viewer as mv
        env = DoorEnv(task=args.task, hand_path=args.hand, door_path=args.door,
                      hand_self_collision=args.self_collision, hand_door_collision=not args.no_hand_door_collision,
                      randomize_physics=args.randomize_physics)
        obs, info = env.reset(seed=0)
        ov = env.hand_overlaps()
        print(f"[view] task {args.task}: palm {info['palm_to_handle'] * 1000:.0f} mm from the handle, "
              f"start moved back {info['start_pushed_back'] * 100:.0f} cm to clear the door, "
              f"overlaps now: {ov if ov else 'none'}")
        print(f"[view] door physics multipliers: {info['physics']}")
        print("[view] holding the start pose; close the window to quit (R in the viewer is NOT reset - rerun)")
        m, d = env.model, env.data
        with mv.launch_passive(m, d) as v:
            v.cam.lookat[:] = env.geo["door"]["grasp"]
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.6, 135, -15
            while v.is_running():
                t0 = time.perf_counter()
                with v.lock():
                    for _ in range(env.frame_skip):
                        mujoco.mj_step(m, d)
                v.sync()
                time.sleep(max(0.0, env.frame_skip * m.opt.timestep - (time.perf_counter() - t0)))
        return
    env = DoorEnv(task=args.task, hand_path=args.hand, door_path=args.door,
                  render_mode=None if args.headless else "human",
                  hand_self_collision=args.self_collision, hand_door_collision=not args.no_hand_door_collision,
                  randomize_physics=args.randomize_physics)
    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep)
        ret, done = 0.0, False
        while not done:
            a = scripted_action(env) if args.scripted else env.action_space.sample()
            obs, r, term, trunc, info = env.step(a)
            ret += r
            done = term or trunc
            if not args.headless:
                import time
                time.sleep(1 / 25)
        print(f"[episode {ep}] steps {env.steps}  return {ret:7.2f}  success {info['success']}  "
              f"palm->handle {info['palm_to_handle'] * 1000:.0f} mm  handle {info['handle_frac'] * 100:.0f} %  "
              f"door {info['door_deg']:.1f} deg")
    env.close()


if __name__ == "__main__":
    main()
