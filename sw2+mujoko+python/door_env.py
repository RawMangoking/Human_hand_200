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
_PRINTED = set()


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


def refresh_body_masks(model):
    """MuJoCo first checks a per-BODY copy of the collision bits (OR of the body's geoms), made when the model is
    compiled. After changing geom_contype / geom_conaffinity at run time that copy must be updated too,
    otherwise e.g. hand<->door pairs are rejected before their geoms are ever compared."""
    if not hasattr(model, "body_contype"):
        return
    ct = np.zeros(model.nbody, dtype=np.int64)
    ca = np.zeros(model.nbody, dtype=np.int64)
    for g in range(model.ngeom):
        b = model.geom_bodyid[g]
        ct[b] |= int(model.geom_contype[g])
        ca[b] |= int(model.geom_conaffinity[g])
    model.body_contype[:] = ct
    model.body_conaffinity[:] = ca


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
    refresh_body_masks(model)
    return n_hand, n_other


def fit_boxes(model, geoms, k=6, n_pts=4000, seed=0):
    """Oriented boxes approximating mesh geoms (body frame): points sampled over the surface (by triangle area),
    k-means into k compact clusters, one PCA-aligned box per cluster. -> [(centre, half-sizes, quat)]"""
    rng = np.random.default_rng(seed)
    pts = []
    for g in geoms:
        mid = model.geom_dataid[g]
        V = np.array(model.mesh_vert[model.mesh_vertadr[mid]: model.mesh_vertadr[mid] + model.mesh_vertnum[mid]])
        F = np.array(model.mesh_face[model.mesh_faceadr[mid]: model.mesh_faceadr[mid] + model.mesh_facenum[mid]])
        A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
        idx = rng.choice(len(F), size=n_pts // len(geoms), p=area / area.sum())
        u, v = rng.random((2, len(idx)))
        flip = u + v > 1
        u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        P = np.vstack([V, A[idx] + u[:, None] * (B[idx] - A[idx]) + v[:, None] * (C[idx] - A[idx])])
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, model.geom_quat[g])
        pts.append(P @ R.reshape(3, 3).T + model.geom_pos[g])          # geom -> body frame
    P = np.vstack(pts)
    k = max(1, min(k, len(P) // 50))
    cent = [P[rng.integers(len(P))]]                                    # farthest-point initialisation
    for _ in range(k - 1):
        dmin = np.min([np.sum((P - c) ** 2, 1) for c in cent], axis=0)
        cent.append(P[int(np.argmax(dmin))])
    cent = np.array(cent)
    for _ in range(30):
        lab = np.argmin(((P[:, None, :] - cent[None]) ** 2).sum(-1), axis=1)
        cent = np.array([P[lab == j].mean(0) if np.any(lab == j) else cent[j] for j in range(k)])
    out = []
    for j in range(k):
        Q = P[lab == j]
        if len(Q) < 10:
            continue
        mu = Q.mean(0)
        _, vec = np.linalg.eigh(np.cov((Q - mu).T))
        if np.linalg.det(vec) < 0:
            vec[:, 0] = -vec[:, 0]
        loc = (Q - mu) @ vec
        lo, hi = loc.min(0), loc.max(0)
        half = np.maximum((hi - lo) / 2, 0.002)
        c = mu + vec @ ((lo + hi) / 2)
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, vec.flatten())
        out.append((c, half, q))
    return out


def build_scene(hand_path, door_path, standoff=0.30, handle_boxes=6):
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
    report = []
    model = mujoco.MjModel.from_xml_string(ET.tostring(scene, encoding="unicode"))

    # (1) door parts the hand must touch (door, handle) need collision shapes: use the visible mesh if none
    for jname in (dcfg["hinge"], dcfg["handle"]):
        b = model.jnt_bodyid[model.joint("door/" + jname).id]
        if not any(model.geom_contype[g] or model.geom_conaffinity[g]
                   for g in range(model.ngeom) if model.geom_bodyid[g] == b):
            bname = model.body(b).name
            el = next(e for e in wb.iter("body") if e.get("name") == bname)
            for g in el.findall("geom"):
                g.set("contype", "1")
                g.set("conaffinity", "1")
            report.append(f"{bname} had no collision shape - its visible mesh now collides")
    if any("collision shape" in r for r in report):
        model = mujoco.MjModel.from_xml_string(ET.tostring(scene, encoding="unicode"))

    # (1b) the handle's collision shape: a convex hull of an L / T-shaped handle (lever + spindle + rod) is one
    #      solid wedge - the fingers would hit empty space. Fit a few oriented boxes to its mesh instead.
    hb = model.jnt_bodyid[model.joint("door/" + dcfg["handle"]).id]
    meshes = [g for g in range(model.ngeom) if model.geom_bodyid[g] == hb and int(model.geom_type[g]) == 7
              and (model.geom_contype[g] or model.geom_conaffinity[g])]
    if meshes and handle_boxes:
        boxes = fit_boxes(model, meshes, k=handle_boxes)
        hel = next(e for e in wb.iter("body") if e.get("name") == model.body(hb).name)
        gx = [c for c in hel if c.tag == "geom"]
        first = model.body_geomadr[hb]
        for g in meshes:                                        # the mesh stays visible, boxes collide
            if 0 <= g - first < len(gx):
                gx[g - first].set("contype", "0")
                gx[g - first].set("conaffinity", "0")
        for i, (c, half, q) in enumerate(boxes):
            ET.SubElement(hel, "geom", name=f"door/handle_box{i}", type="box", size=_fmt(half), pos=_fmt(c),
                          quat=_fmt(q), contype="1", conaffinity="1", group="3", rgba="0.2 0.6 1 0.4")
        report.append(f"handle collision: {len(boxes)} boxes fitted to its shape (instead of one convex hull)")
        model = mujoco.MjModel.from_xml_string(ET.tostring(scene, encoding="unicode"))

    # (2) door parts that overlap each other at rest (e.g. the handle sitting inside the frame in CAD) would be
    #     shoved apart at the first step and fling the door: switch that pair off (the env latch keeps the door
    #     shut until the handle is turned)
    d0 = mujoco.MjData(model)
    mujoco.mj_forward(model, d0)
    hand_b = {b for b in range(model.nbody) if model.body(b).name.startswith("hand/")}
    pairs = {}
    for c in d0.contact[:d0.ncon]:
        b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
        if b1 in hand_b or b2 in hand_b or c.dist > -0.001:
            continue
        key = tuple(sorted((model.body(b1).name, model.body(b2).name)))
        pairs[key] = max(pairs.get(key, 0.0), -float(c.dist))
    if pairs:
        cont = scene.find("contact")
        if cont is None:
            cont = ET.SubElement(scene, "contact")
        for (n1, n2), depth in sorted(pairs.items()):
            ET.SubElement(cont, "exclude", body1=n1, body2=n2)
            report.append(f"{n1} overlaps {n2} by {depth * 1000:.1f} mm at rest - that pair's collision is OFF "
                          f"(fix in CAD to restore it; the env latch still holds the door shut)")
        model = mujoco.MjModel.from_xml_string(ET.tostring(scene, encoding="unicode"))
    for r in report:
        if r not in _PRINTED:                                   # say each scene fix once per run
            _PRINTED.add(r)
            print("[door_env] " + r)

    xml = ET.tostring(scene, encoding="unicode")
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(door_path)))),
                           "Hand_Door_env")
    try:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "scene.xml"), "w") as f:
            f.write(xml)
    except OSError:
        pass
    return model, hcfg, dcfg, dict(hand=hg, door=dg, root_pos=root_pos, root_quat=root_quat, report=report)


# --------------------------------------------------------------------------- test doors
UNSEEN_DOORS = [                                   # every one is OUTSIDE the training ranges
    ("shifted_side", {"door_dx": 0.14}),                 # door 14 cm to the side (train: +-8 cm)
    ("handle_high", {"door_dz": 0.10}),                  # handle 10 cm higher (train: +-6 cm)
    ("handle_low", {"door_dz": -0.10}),
    ("turned_door", {"door_yaw": 18.0}),                 # door turned 18 deg (train: +-10 deg)
    ("stiff_handle", {"handle_friction": 15.0}),         # 15x handle resistance (train: up to 10x)
    ("stiff_hinge", {"hinge_friction": 15.0, "hinge_damping": 3.0}),
    ("deep_latch", {"unlock": 0.95}),                    # handle must turn 95 % (train: 60-90 %)
    ("heavy_slippery", {"door_mass": 2.0, "grip_friction": 0.4}),
    ("hard_combo", {"door_dx": 0.12, "door_dz": 0.08, "door_yaw": -15.0, "handle_friction": 12.0}),
]


def door_set(name, n=10, seed=1234):
    """Named sets of test doors -> [(label, physics multipliers)].
    nominal : the training door of the fixed condition
    in_dist : n doors sampled inside the training ranges (fixed seed -> same doors every time)
    unseen  : UNSEEN_DOORS, outside the training ranges (generalization test)"""
    if name == "nominal":
        return [("nominal", {})]
    if name == "in_dist":
        rng = np.random.default_rng(seed)
        return [(f"in_dist_{i}", {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in DoorEnv.PHYSICS_RANGES.items()})
                for i in range(n)]
    if name == "unseen":
        return list(UNSEEN_DOORS)
    if name == "all":
        return door_set("nominal") + door_set("in_dist", n, seed) + door_set("unseen")
    raise ValueError(f"unknown door set '{name}' (nominal, in_dist, unseen, all)")


# --------------------------------------------------------------------------- environment
class DoorEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}
    TASKS = {                                   # action channels per task, action_mode="synergy" (default)
        "reach": ["arm_pos", "arm_rot"],
        "handle": ["arm_pos", "arm_rot", "wrist", "index", "mrp", "thumb", "opp"],
        "door": ["arm_pos", "grip"],
        "full": ["arm_pos", "arm_rot", "wrist", "index", "mrp", "thumb", "opp"],   # one policy, whole sequence
    }
    TASKS_FULL = {                              # action_mode="full": every hand joint is its own action
        "reach": ["arm_pos", "arm_rot"],
        "handle": ["arm_pos", "arm_rot", "joints"],
        "door": ["arm_pos", "arm_rot", "joints"],
        "full": ["arm_pos", "arm_rot", "joints"],
    }
    MAX_STEPS = {"reach": 250, "handle": 250, "door": 250, "full": 600}
    SIZES = {"arm_pos": 3, "arm_rot": 3, "wrist": 2, "index": 1, "mrp": 1, "thumb": 1, "opp": 1, "grip": 1}
    LEASH = 0.02             # m: how far the arm target may run ahead of the arm
    WORKSPACE = 0.70         # m: the arm stays within this distance of where it started (a real arm's reach)
    STEP_FRAC = 0.10         # finger / wrist actions: fraction of the range per step at action = 1
    # domain randomization: each value is a multiplier of the door's nominal value (from door_setup.py)
    # multipliers of the nominal door (from door_setup.py) ...
    PHYSICS_RANGES = {"door_mass": (0.6, 1.4), "hinge_damping": (0.5, 2.0), "hinge_friction": (0.5, 10.0),
                      "handle_friction": (0.5, 10.0), "grip_friction": (0.6, 1.4),
                      # ... and absolute values: where the door is, how far the latch must be turned
                      "door_dx": (-0.08, 0.08),      # m, sideways (along the door face)
                      "door_dz": (-0.06, 0.06),      # m, up / down
                      "door_yaw": (-10.0, 10.0),     # deg, door turned about the vertical
                      "unlock": (0.6, 0.9)}          # handle travel (fraction) that releases the latch
    NOMINAL = {"door_mass": 1.0, "hinge_damping": 1.0, "hinge_friction": 1.0, "handle_friction": 1.0,
               "grip_friction": 1.0, "door_dx": 0.0, "door_dz": 0.0, "door_yaw": 0.0, "unlock": 0.8}

    def __init__(self, task="reach", hand_path=DEFAULT_HAND, door_path=DEFAULT_DOOR, render_mode=None,
                 control_hz=25, max_steps=None, door_dir="push", randomize=True, latch=True, unlock_frac=0.8,
                 hand_self_collision=False, hand_door_collision=True,
                 randomize_physics=False, physics=None, physics_ranges=None,
                 action_mode="synergy", obs_noise=0.0, start_states=None):
        assert task in self.TASKS, f"task must be one of {list(self.TASKS)}"
        assert action_mode in ("synergy", "full"), "action_mode must be 'synergy' or 'full'"
        self.task, self.render_mode, self.randomize = task, render_mode, randomize
        self.action_mode, self.obs_noise, self.start_states = action_mode, float(obs_noise), start_states
        self._fixed_max_steps = max_steps
        self.base_task = task                                   # reset() always returns to this task
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
            if "nohand" not in _PRINTED:
                _PRINTED.add("nohand")
                print(f"[door_env] the hand had NO collision shapes - using its {n_hand} visible meshes for collision")
            refresh_body_masks(self.model)
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
        self.max_steps = max_steps or self.MAX_STEPS[task]
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
        self._latched = True
        m.jnt_solref[self.jh] = [0.005, 1.0]                     # a firm latch / door stop

        # arm (mocap target)
        self.mid = m.body_mocapid[B("hand/base_target")]
        self.free_q = m.jnt_qposadr[J("hand/base_free")]
        self.free_v = m.jnt_dofadr[J("hand/base_free")]

        self.joint_names = list(self.hj)
        self.sizes = dict(self.SIZES, joints=len(self.joint_names))
        self._target = {n: self._home(n) for n in self.hj}
        self._curl = {g: 0.0 for g in ("index", "mrp", "thumb", "opp")}
        self._wpos = [0.0 for _ in self.wrist]
        self._stage = 0
        self._set_spaces(task)
        self._viewer, self._renderer = None, None
        self.steps = 0
        self.hand_door_collision = hand_door_collision

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
        self.NOMINAL = dict(self.NOMINAL, unlock=unlock_frac)
        self.physics = dict(self.NOMINAL)
        b = self.door_body
        while m.body_parentid[b] != 0:
            b = m.body_parentid[b]
        self.door_root = b
        self._root_pos0, self._root_quat0 = m.body_pos[b].copy(), m.body_quat[b].copy()
        self._grasp0 = self.geo["door"]["grasp"].copy()
        self._normal0 = self.geo["door"]["normal"].copy()
        self._door_yaw = 0.0

    # ---- task / action layout
    def channels(self, task=None):
        t = task or self.task
        return (self.TASKS_FULL if self.action_mode == "full" else self.TASKS)[t]

    def _set_spaces(self, task):
        self.task = task
        self.action_layout = [(k, self.sizes[k]) for k in self.channels(task)]
        n_act = sum(n for _, n in self.action_layout)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_act,), dtype=np.float32)
        self._last_action = np.zeros(n_act, dtype=np.float32)
        self.observation_space = spaces.Box(-1e3, 1e3, shape=self._obs().shape, dtype=np.float32)
        if self._fixed_max_steps is None:
            self.max_steps = self.MAX_STEPS[task] if hasattr(self, "frame_skip") else 250

    def switch_task(self, task):
        """Change task (action set, observation, reward) WITHOUT touching the simulation - for chaining the
        three modular policies on one door: reach -> handle -> door."""
        self._set_spaces(task)
        self.steps = 0
        self._prev = self._progress()
        return self._obs()

    # ---- simulation state (for start-state pools: start 'handle' where 'reach' ended, etc.)
    def get_state(self):
        d = self.data
        return dict(qpos=d.qpos.copy(), qvel=d.qvel.copy(), ctrl=d.ctrl.copy(), act=d.act.copy(),
                    mocap_pos=d.mocap_pos.copy(), mocap_quat=d.mocap_quat.copy(), target=dict(self._target),
                    curl=dict(self._curl), wpos=list(self._wpos), arm_start=self._arm_start.copy(),
                    physics=dict(self.physics), latched=self._latched)

    def set_state(self, st):
        m, d = self.model, self.data
        d.qpos[:], d.qvel[:], d.ctrl[:] = st["qpos"], st["qvel"], st["ctrl"]
        if m.na:
            d.act[:] = st["act"]
        d.mocap_pos[:], d.mocap_quat[:] = st["mocap_pos"], st["mocap_quat"]
        self._target, self._curl, self._wpos = dict(st["target"]), dict(st["curl"]), list(st["wpos"])
        self._latched = st.get("latched", True)
        self._update_latch()
        mujoco.mj_forward(m, d)

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

    def _place_arm(self, palm_offset, yaw, at_handle=True):
        """Put the mocap target AND the free joint so the palm is `palm_offset` from a handle.
        at_handle=True : relative to the ACTUAL handle (the previous stage ended there: handle / door tasks)
        at_handle=False: relative to the NOMINAL handle - the robot stands at a fixed spot in the world and a
                         moved / turned door really is somewhere else (reach / full tasks)"""
        g = self.geo
        R0 = np.zeros(9)
        mujoco.mju_quat2Mat(R0, g["root_quat"])
        if at_handle:
            yaw = yaw + self._door_yaw                                  # face the (turned) door
        Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        R = Rz @ R0.reshape(3, 3)
        palm0 = self._grasp0 + self._normal0 * 0.30                    # where build_scene put the palm
        base = g["door"]["grasp"] if at_handle else self._grasp0
        pos = base + palm_offset + Rz @ (g["root_pos"] - palm0)
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
        if self.obs_noise > 0 and hasattr(self, "np_random"):
            n = len(o) - len(self._last_action)                 # noise on sensed values, not on own action
            o[:n] = o[:n] + self.np_random.normal(0.0, self.obs_noise, n)
        return np.clip(o, -1e3, 1e3).astype(np.float32)

    # ---- door physics (domain randomization)
    def _apply_physics(self, mult):
        mult = dict(self.NOMINAL, **mult)
        m, n = self.model, self._nominal
        # door pose: turn about the vertical through the handle, then shift along the door face / up
        yaw = math.radians(mult["door_yaw"])
        c, s_ = math.cos(yaw), math.sin(yaw)
        Rz = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
        side = np.cross(self._normal0, [0.0, 0.0, 1.0])
        side /= np.linalg.norm(side) + 1e-12
        m.body_pos[self.door_root] = (self._grasp0 + Rz @ (self._root_pos0 - self._grasp0)
                                      + side * mult["door_dx"] + np.array([0, 0, mult["door_dz"]]))
        qz = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, qz, self._root_quat0)
        m.body_quat[self.door_root] = q
        self._door_yaw = yaw
        self.unlock_frac = float(mult["unlock"])
        m.body_mass[self.door_body] = n["door_mass"] * mult["door_mass"]
        m.body_inertia[self.door_body] = n["door_inertia"] * mult["door_mass"]
        m.dof_damping[self.dof_h] = n["hinge_damping"] * mult["hinge_damping"]
        m.dof_frictionloss[self.dof_h] = n["hinge_friction"] * mult["hinge_friction"]
        m.dof_frictionloss[self.dof_d] = n["handle_friction"] * mult["handle_friction"]
        m.geom_friction[self.handle_geoms, 0] = n["grip_friction"] * mult["grip_friction"]
        self.physics = dict(mult)
        # where the handle / door face are now (used for placing the arm and by the scripted policies)
        d = self.data                                   # (door joints are at rest: reset() just reset them)
        mujoco.mj_forward(m, d)
        self.geo["door"]["grasp"] = np.array(d.xipos[self.handle_body])
        self.geo["door"]["normal"] = Rz @ self._normal0

    def _sample_physics(self):
        if self.fixed_physics is not None:
            mult = dict(self.NOMINAL, **self.fixed_physics)
        elif self.randomize_physics:
            mult = {k: float(self.np_random.uniform(lo, hi)) for k, (lo, hi) in self.physics_ranges.items()}
        else:
            mult = dict(self.NOMINAL)
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

    def _place_clear(self, normal_offset, lateral, yaw, at_handle=True):
        """Place the palm `normal_offset` in front of the handle (+ lateral offset); if any hand part overlaps
        the door/frame/handle/floor, move the arm back along the door normal 1 cm at a time until it is clear."""
        nrm = self.geo["door"]["normal"] if at_handle else self._normal0
        pushed = 0.0
        for _ in range(120):
            self._place_arm(nrm * (normal_offset + pushed) + lateral, yaw, at_handle)
            mujoco.mj_forward(self.model, self.data)
            if not self.hand_door_collision or not self.hand_overlaps():
                break
            pushed += 0.01
        self.start_pushed_back = pushed
        return pushed

    # ---- gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.task != self.base_task:                         # after chaining (switch_task), start over
            self._set_spaces(self.base_task)
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        self._target = {n: self._home(n) for n in self.hj}
        self._curl = {g: 0.0 for g in ("index", "mrp", "thumb", "opp")}
        self._wpos = [0.0 for _ in self.wrist]
        for n, h in self.hj.items():
            d.qpos[h["qadr"]] = self._target[n]
            d.ctrl[h["act"]] = self._target[n]
        rnd = self.np_random
        if options and "physics" in options:                   # evaluate on a specific test door
            self._apply_physics(dict(self.NOMINAL, **options["physics"]))
        else:
            self._sample_physics()
        noise = (rnd.uniform(-1, 1, 3) * 0.05) if self.randomize else np.zeros(3)
        yaw = rnd.uniform(-1, 1) * math.radians(10) if self.randomize else 0.0
        at_handle = self.task in ("handle", "door")
        nrm = self.geo["door"]["normal"] if at_handle else self._normal0
        lateral = noise - nrm * (noise @ nrm)                   # sideways / vertical part of the noise
        along = float(noise @ nrm)
        if self.task == "door":                                 # door: handle already turned open
            d.qpos[self.qd] = self.handle_open
            if self.latch_eq >= 0:
                d.eq_active[self.latch_eq] = 0
        start = {"reach": (0.30 + along, lateral, yaw),
                 "full": (0.30 + along, lateral, yaw),
                 "handle": (0.07 + 0.3 * along, 0.3 * lateral, 0.3 * yaw),
                 "door": (0.05 + 0.3 * along, 0.3 * lateral, 0.3 * yaw)}[self.task]
        self._latched = self.task != "door"                     # door task: the handle is already turned
        self._place_clear(*start, at_handle=at_handle)
        self._arm_start = self.data.mocap_pos[self.mid].copy()
        self._stage = 0
        if self.start_states:                                   # start where a previous policy ended
            self.set_state(self.start_states[int(rnd.integers(len(self.start_states)))])
            self.start_pushed_back = 0.0
            self._arm_start = st0 if (st0 := self.start_states[0].get("arm_start")) is not None else \
                self.data.mocap_pos[self.mid].copy()
        self._update_latch()
        mujoco.mj_forward(m, d)
        self.steps = 0
        self._last_action[:] = 0
        self._prev = self._progress()
        return self._obs(), self._info()

    def _apply_action(self, a):
        d = self.data
        k = 0
        for name, n in self.action_layout:
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
            elif name == "joints":                              # full mode: one action per hand joint
                for jn, val in zip(self.joint_names, x):
                    lo, hi = self.hj[jn]["cfg"]["range"]
                    self._target[jn] = float(np.clip(self._target[jn] + self.STEP_FRAC * (hi - lo) * float(val),
                                                     lo, hi))
        # leash: the target never runs more than LEASH ahead of the real arm. With the soft weld this caps the
        # arm's push at ~20 N, so it cannot force the hand through the door (a real arm is force-limited too)
        arm = d.qpos[self.free_q:self.free_q + 3]
        off = d.mocap_pos[self.mid] - arm
        n = np.linalg.norm(off)
        if n > self.LEASH:
            d.mocap_pos[self.mid] = arm + off * self.LEASH / n
        ws = d.mocap_pos[self.mid] - self._arm_start                    # workspace limit
        wn = np.linalg.norm(ws)
        if wn > self.WORKSPACE:
            d.mocap_pos[self.mid] = self._arm_start + ws * self.WORKSPACE / wn

    def _update_latch(self):
        """A real latch: it stays latched until the handle is turned unlock_frac of its travel - however hard
        the door is pushed - and re-latches only when the door is back near closed with the handle released."""
        m, d = self.model, self.data
        if not self.latch:
            return
        frac = (d.qpos[self.qd] - self.d0) / (self.handle_open - self.d0 + 1e-9)
        closed = abs(d.qpos[self.qh] - self.h0) < math.radians(1.5)
        if self._latched and frac >= self.unlock_frac:
            self._latched = False
        elif not self._latched and closed and frac < self.unlock_frac:
            self._latched = True
        if self._latched:
            m.jnt_range[self.jh] = [self.h0 - 1e-3, self.h0 + 1e-3]
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
        elif self.task == "door":
            r = 5.0 * (p["door"] - self._prev["door"]) + 0.02 * touch
            success = p["door"] > math.radians(60)
        else:                                                   # full: staged reward, one policy does it all
            r = 0.2 * (-p["dist"] + 5.0 * (self._prev["dist"] - p["dist"]))
            r += 10.0 * (p["handle"] - self._prev["handle"]) + 0.05 * len(tips)
            r += 5.0 * (p["door"] - self._prev["door"])
            if self._stage == 0 and p["dist"] < 0.04:
                self._stage, r = 1, r + 2.0                     # reached the handle
            if self._stage <= 1 and p["handle"] > self.unlock_frac:
                self._stage, r = 2, r + 5.0                     # handle turned
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
        return dict(palm_to_handle=p["dist"], handle_frac=p["handle"], task=self.task, stage=self._stage,
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
def named_action(env, **channels):
    """Build an action vector from named channels, e.g. named_action(env, arm_pos=[0,0,1], mrp=1)."""
    a = np.zeros(env.action_space.shape, dtype=np.float32)
    k = 0
    for name, n in env.action_layout:
        if name in channels:
            a[k:k + n] = channels[name]
        k += n
    return a


def _close(env, amount):
    """Finger-closing channels for either action mode (synergy curls or every flex joint)."""
    names = [n for n, _ in env.action_layout]
    if "joints" in names:
        flex = np.array([1.0 if env.hj[j]["cfg"]["role"] == "flex" else 0.0 for j in env.joint_names])
        return {"joints": flex * amount}
    if "grip" in names:
        return {"grip": amount}
    return {k: amount for k in ("index", "mrp", "thumb", "opp") if k in names}


def scripted_action(env):
    """Hand-written policies for every task (shows each task is solvable; a baseline; used by the tests).
    For task 'full' it runs the three stages in order."""
    d = env.data
    palm_c = np.array(d.xipos[env.palm])
    grasp = np.array(d.xipos[env.handle_body])
    to = np.clip((grasp - palm_c) / 0.01, -1, 1)
    near = np.linalg.norm(grasp - palm_c) < 0.05
    stage = {"reach": 0, "handle": 1, "door": 2}.get(env.task, env._stage)
    if env.task == "full" and stage == 0 and near:
        stage = 1
    if stage == 0:
        return named_action(env, arm_pos=to, **_close(env, -1.0))
    if stage == 1:
        if not near:
            return named_action(env, arm_pos=to, **_close(env, -1.0))
        ax, piv = np.array(d.xaxis[env.jd]), np.array(d.xanchor[env.jd])
        tang = np.cross(ax, grasp - piv) * np.sign(env.handle_open - env.d0)
        n = np.linalg.norm(tang)
        move = np.clip(tang / (n + 1e-9) + 0.5 * to, -1, 1) if n > 1e-6 else to
        return named_action(env, arm_pos=move, **_close(env, 1.0))
    # push perpendicular to the door's CURRENT face (it turns as the door swings)
    ang = float(d.qpos[env.qh] - env.h0)
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.array(d.xaxis[env.jh]), ang)
    nrm_now = np.zeros(3)
    mujoco.mju_rotVecQuat(nrm_now, env.geo["door"]["normal"], q)
    push = -nrm_now * env.door_goal_sign * env.geo["door"]["push_sign"]
    door_c = np.array(d.xipos[env.door_body])                       # and keep the hand near the door
    keep = np.clip((door_c - palm_c) / 0.3, -1, 1)
    keep -= nrm_now * (keep @ nrm_now)
    return named_action(env, arm_pos=np.clip(push + 0.3 * keep, -1, 1), **_close(env, -1.0))


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
