#!/usr/bin/env python3
"""
add_actuators.py
================
Post-processes the MJCF that sw2robot exports
(python -m sw2robot.exporter.export Hand_forearm.SLDASM -o output --mujoco --mujoco-fixed-base)
and turns it into a fully actuated, controllable hand:

  * one POSITION actuator per hinge/slide joint (replaces whatever actuators sw2robot wrote)
  * kp sized from the load each joint actually carries (subtree mass x lever arm),
    damping for a critically-damped response, armature for numerical stability
  * ranges filled in for any joint that came out unlimited (with a warning)
  * optional extra axes on an existing joint (finger abduction, 2-DOF thumb CMC ...)
    via --add-axis, because MuJoCo allows several hinges in one body
  * contacts that already touch in the open pose are excluded (stops start-up jitter)
  * a hand_config.json that hand_controller.py uses to map the gamepad onto joints

Joints are grouped automatically: the palm is found as the body the fingers hang off,
every chain below it is named by the earliest finger keyword in its link names
(thumb / index / middle / ring / pinky|little), and joints above the palm become "wrist".

The output is written NEXT TO the input so relative mesh paths keep working.

Usage
  python add_actuators.py path/to/robot.xml
  python add_actuators.py path/to/robot.xml --weld-base            # if you exported without --mujoco-fixed-base
  python add_actuators.py path/to/robot.xml --point-up             # stand the arm up: forearm -> palm along +Z
  python add_actuators.py path/to/robot.xml --add-axis Index_Proximal_joint   # auto abduction axis
  python add_actuators.py path/to/robot.xml --add-axis "Thumb_Base_joint:0,1,0"  # explicit axis (body frame)
"""
import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

FINGER_KEYWORDS = {
    "thumb": ("thumb",),
    "index": ("index",),
    "middle": ("middle",),
    "ring": ("ring",),
    "pinky": ("pinky", "little"),
}
FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]

HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)
SLIDE = int(mujoco.mjtJoint.mjJNT_SLIDE)
BALL = int(mujoco.mjtJoint.mjJNT_BALL)
FREE = int(mujoco.mjtJoint.mjJNT_FREE)


# --------------------------------------------------------------------------- helpers
def fmt(*vals):
    return " ".join(f"{float(v):.6g}" for v in vals)


def finger_of(name):
    """Finger whose keyword appears EARLIEST in the name.
    'Index_Middle-1' -> index (not middle), 'Pinky_Middle' -> pinky."""
    low = (name or "").lower()
    best, best_pos = None, len(low) + 1
    for finger, keys in FINGER_KEYWORDS.items():
        for k in keys:
            p = low.find(k)
            if p != -1 and p < best_pos:
                best, best_pos = finger, p
    return best


def compile_tree(tree, workdir):
    """Compile the XML tree from a temp file in the model folder (keeps mesh paths valid)."""
    tmp = os.path.join(workdir, "__add_actuators_tmp.xml")
    tree.write(tmp)
    try:
        return mujoco.MjModel.from_xml_path(tmp)
    finally:
        os.remove(tmp)


def world_joints(root):
    wb = root.find("worldbody")
    return [] if wb is None else list(wb.iter("joint"))


def find_joint_elem(root, name):
    for j in world_joints(root):
        if j.get("name") == name:
            return j
    return None


def parent_map(root):
    return {c: p for p in root.iter() for c in p}


def resolve_body(m, key):
    """Body whose name equals / contains key (case-insensitive, must be unique)."""
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, key)
    if b >= 0:
        return b
    hits = [i for i in range(1, m.nbody) if key.lower() in m.body(i).name.lower()]
    if len(hits) != 1:
        sys.exit(f"'{key}' matches {len(hits)} bodies: {[m.body(i).name for i in hits]}")
    return hits[0]


def resolve_joint(m, key):
    """Hinge joint by exact name, by part of its name, or by part of its body's name."""
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, key)
    if j >= 0:
        return j
    hinges = [i for i in range(m.njnt) if int(m.jnt_type[i]) in (HINGE, SLIDE)]
    hits = [i for i in hinges if key.lower() in m.joint(i).name.lower()]
    if len(hits) != 1:
        bodies = {int(m.jnt_bodyid[i]) for i in hinges if key.lower() in m.body(int(m.jnt_bodyid[i])).name.lower()}
        if len(bodies) == 1:
            b = bodies.pop()
            return min(i for i in hinges if m.jnt_bodyid[i] == b)
        sys.exit(f"'{key}' matches {len(hits)} joints / {len(bodies)} bodies - use a more specific name. "
                 f"Joints: {[m.joint(i).name for i in hinges]}")
    return hits[0]


def drop_keyframes(root):
    """Keyframe qpos no longer matches once joints are added/removed."""
    for kf in root.findall("keyframe"):
        root.remove(kf)


def get_or_make(root, tag):
    el = root.find(tag)
    return el if el is not None else ET.SubElement(root, tag)


# --------------------------------------------------------------------------- analysis
def analyse(m, extra_names):
    """Find palm, group joints into wrist / fingers, order them proximal -> distal."""
    nb = m.nbody
    parent = m.body_parentid
    children = [[] for _ in range(nb)]
    depth = [0] * nb
    for b in range(1, nb):
        children[parent[b]].append(b)
        depth[b] = depth[parent[b]] + 1

    named_palm = [b for b in range(1, nb)
                  if "palm" in m.body(b).name.lower() and len(children[b]) >= 2]
    palm = named_palm[0] if named_palm else max(range(1, nb), key=lambda b: len(children[b]))

    def ancestors(b):
        out = []
        while b != 0:
            out.append(b)
            b = parent[b]
        return out

    joints, skipped = {}, []
    chain_group = {}
    for j in range(m.njnt):
        jt = int(m.jnt_type[j])
        name = m.joint(j).name
        if jt in (FREE, BALL):
            skipped.append((name, "free" if jt == FREE else "ball"))
            continue
        b = int(m.jnt_bodyid[j])
        anc = ancestors(b)
        if palm in anc and b != palm:
            root_b = anc[anc.index(palm) - 1]          # body directly under the palm
            if root_b not in chain_group:
                g = (finger_of(m.body(root_b).name) or finger_of(name)
                     or finger_of(m.body(b).name) or f"digit{len(chain_group) + 1}")
                if g in chain_group.values():         # two chains matched the same keyword
                    g = f"{g}_{sum(v.startswith(g) for v in chain_group.values()) + 1}"
                chain_group[root_b] = g
            group = chain_group[root_b]
            role = "spread" if name in extra_names else "flex"
        else:
            group, role = "wrist", "wrist"
        joints[name] = dict(id=j, body=b, depth=depth[b], group=group, role=role, type=jt)

    fingers = {}
    for n, info in joints.items():
        if info["role"] == "flex":
            fingers.setdefault(info["group"], []).append(n)
    for g in fingers:
        fingers[g].sort(key=lambda n: (joints[n]["depth"], joints[n]["id"]))
    fingers = dict(sorted(fingers.items(),
                          key=lambda kv: FINGER_ORDER.index(kv[0]) if kv[0] in FINGER_ORDER else 99))
    wrist = sorted([n for n, i in joints.items() if i["group"] == "wrist"],
                   key=lambda n: (joints[n]["depth"], joints[n]["id"]))
    spread = [n for n, i in joints.items() if i["role"] == "spread"]
    # wrist roles: added axis = deviation (tilt), deepest original = flexion, the rest = twist
    wrist_axes = {}
    base = [n for n in wrist if n not in extra_names]
    extras = [n for n in wrist if n in extra_names]
    if base:
        wrist_axes["flex"] = base[-1]
    if len(base) == 2 and not extras:
        wrist_axes["deviation"] = base[0]            # 2-DOF wrist: tilt + flexion (both CAD hinges)
    else:
        for n in base[:-1]:
            wrist_axes.setdefault("twist", n)
    for n in extras:
        wrist_axes.setdefault("deviation", n)
    return palm, joints, fingers, wrist, spread, skipped, wrist_axes


# --------------------------------------------------------------------------- anatomical joint frames
def anatomical_setup(root, m, palm, fingers, wrist_axes, flex_lo, flex_hi, fix_wrist, ang,
                     cmc_lo=0.0, cmc_hi=math.pi / 2, straight_tol=math.radians(10)):
    """Make every finger/thumb flexion joint read as an anatomical angle:
         * axis flipped where needed so that POSITIVE = flexion (toward the palm)
         * ref = the bend the CAD pose already has  ->  q = 0 means STRAIGHT
         * range = [flex_lo, flex_hi]
       The bend is measured between the parent bone (previous joint -> this joint)
       and the child bone (this joint -> next joint / fingertip centre of mass).
       Optionally rebuild the two wrist axes from the hand: flexion parallel to the
       knuckle axes, deviation (tilt) perpendicular to the knuckles and the hand length."""
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)

    def body_anchor(b):
        for j in range(m.njnt):
            if m.jnt_bodyid[j] == b and int(m.jnt_type[j]) == HINGE:
                return np.array(d.xanchor[j])
        return np.array(d.xipos[b])

    rows, info, confident = [], {}, []
    cmc = None
    for g, names in fingers.items():
        for k, n in enumerate(names):
            j = jid(n)
            b = m.jnt_bodyid[j]
            a = np.array(d.xaxis[j])
            p0 = np.array(d.xanchor[j])
            if g.startswith("thumb") and k == 0:
                cmc = (n, j, a, p0, np.array(d.xanchor[jid(names[1])]) if len(names) > 1
                       else np.array(d.subtree_com[b]))
                continue
            par = body_anchor(m.body_parentid[b])
            tip = np.array(d.xanchor[jid(names[k + 1])]) if k + 1 < len(names) else np.array(d.subtree_com[b])
            pd, cd = p0 - par, tip - p0
            pp, cp = pd - a * (pd @ a), cd - a * (cd @ a)
            if np.linalg.norm(pp) < 0.4 * np.linalg.norm(pd) or np.linalg.norm(cp) < 0.4 * np.linalg.norm(cd):
                rows.append((n, "kept (bone runs along the axis)"))
                continue
            beta = math.atan2(float(a @ np.cross(pp, cp)), float(pp @ cp))
            info[n] = dict(j=j, a=a, r=tip - p0, beta=beta)
            if abs(beta) > math.radians(8):
                info[n]["s"] = 1.0 if beta > 0 else -1.0
                confident.append(n)

    # joints that are (almost) straight in CAD: pick the sign that moves them to the palm side
    if confident:
        P = np.mean([unit3(np.cross(info[n]["s"] * info[n]["a"], info[n]["r"])) for n in confident], axis=0)
    for n, v in info.items():
        if "s" not in v:
            v["s"] = 1.0 if (not confident or float(np.cross(v["a"], v["r"]) @ P) >= 0) else -1.0

    for n, v in info.items():
        s, j = v["s"], v["j"]
        phi0 = s * v["beta"]                       # bend already present in the CAD pose (flexion +)
        snapped = abs(phi0) < straight_tol
        if snapped:                                # nearly straight in CAD (e.g. faces concentric):
            phi0 = 0.0                             # the CAD pose IS the straight pose - don't re-zero
        lo, hi = min(flex_lo, phi0), max(flex_hi, phi0)
        el = find_joint_elem(root, n)
        el.set("axis", fmt(*(s * np.array(m.jnt_axis[j]))))
        el.set("ref", fmt(ang(phi0)))
        el.set("range", fmt(ang(lo), ang(hi)))
        el.set("limited", "true")
        bend_txt = (f"CAD bend {math.degrees(s * v['beta']):5.1f} deg < {math.degrees(straight_tol):.0f} "
                    f"-> CAD pose kept as straight" if snapped else
                    f"CAD bend {math.degrees(phi0):6.1f} deg -> 0 = straight")
        rows.append((n, f"{'axis flipped, ' if s < 0 else ''}{bend_txt}, range "
                        f"[{math.degrees(lo):.0f}, {math.degrees(hi):.0f}]"))

    # thumb CMC: its axis runs along the hand, so measure the thumb against the PALM PLANE
    # (the plane of the hand length and the knuckle line): 0 = in the palm plane, + = away from it
    if cmc is not None:
        n, j, a, p0, tip = cmc
        mcps = [nm[0] for gg, nm in fingers.items() if not gg.startswith("thumb") and nm[0] in info]
        if mcps:
            K = np.sum([info[q]["s"] * info[q]["a"] for q in mcps], axis=0)
            e, c = K - a * (K @ a), (tip - p0) - a * ((tip - p0) @ a)
            if np.linalg.norm(e) > 1e-9 and np.linalg.norm(c) > 1e-9:
                if e @ c < 0:
                    e = -e
                beta = math.atan2(float(a @ np.cross(e, c)), float(e @ c))
                s_ = 1.0 if beta >= 0 else -1.0
                phi0 = abs(beta)
                lo, hi = min(cmc_lo, phi0), max(cmc_hi, phi0)
                el = find_joint_elem(root, n)
                el.set("axis", fmt(*(s_ * np.array(m.jnt_axis[j]))))
                el.set("ref", fmt(ang(phi0)))
                el.set("range", fmt(ang(lo), ang(hi)))
                el.set("limited", "true")
                rows.insert(0, (n, f"{'axis flipped, ' if s_ < 0 else ''}CMC: {math.degrees(phi0):5.1f} deg from "
                                   f"the palm plane in CAD -> 0 = in palm plane, range [{math.degrees(lo):.0f}, "
                                   f"{math.degrees(hi):.0f}]"))
            else:
                rows.insert(0, (n, "kept (thumb CMC: could not measure against the palm plane)"))
        else:
            rows.insert(0, (n, "kept (thumb CMC: no finger MCPs to define the palm plane)"))

    print("\n[anatomical] finger / thumb joints")
    for n, txt in rows:
        print(f"   {n:<44} {txt}")

    # wrist: rebuild axes from the hand geometry
    if fix_wrist and "flex" in wrist_axes and "deviation" in wrist_axes:
        mcps = [names[0] for g, names in fingers.items() if not g.startswith("thumb") and names[0] in info]
        if len(mcps) >= 2:
            K = unit3(np.sum([info[n]["s"] * info[n]["a"] for n in mcps], axis=0))
            A = np.mean([d.xanchor[jid(n)] for n in mcps], axis=0)
            wf, wd = jid(wrist_axes["flex"]), jid(wrist_axes["deviation"])
            L = unit3(A - np.array(d.xanchor[wf]))
            Kp = unit3(K - L * (K @ L))
            N = unit3(np.cross(L, Kp))
            print("\n[wrist] current axes vs the hand (0 deg = parallel):")
            for role, jj in (("flex", wf), ("deviation", wd)):
                ax = np.array(d.xaxis[jj])
                dg = lambda v: math.degrees(math.acos(min(1.0, abs(float(ax @ v)))))
                print(f"   {role:<10} {m.joint(jj).name:<36} to knuckles {dg(Kp):5.1f}  "
                      f"to palm normal {dg(N):5.1f}  to hand length {dg(L):5.1f}")
            for jj, world in ((wf, Kp), (wd, N)):
                R = np.array(d.xmat[m.jnt_bodyid[jj]]).reshape(3, 3)
                find_joint_elem(root, m.joint(jj).name).set("axis", fmt(*(R.T @ world)))
            c = np.array(d.xipos[m.jnt_bodyid[wd]])
            off = max(np.linalg.norm(np.array(d.xanchor[wf]) - c), np.linalg.norm(np.array(d.xanchor[wd]) - c))
            print(f"   -> set: flexion parallel to the knuckles, tilt perpendicular to palm plane "
                  f"(pivots are {off * 1000:.1f} mm from the wrist-ball centre)")
        else:
            print("[wrist] not enough finger MCP joints to rebuild the wrist axes - skipped")


def unit3(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


# --------------------------------------------------------------------------- orientation
def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])


def quat_from_to(u, v):
    """Shortest rotation taking unit vector u onto unit vector v (w, x, y, z)."""
    u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
    c = float(u @ v)
    if c < -0.999999:                                   # opposite: 180 deg about any perpendicular
        ax = np.cross(u, [1, 0, 0])
        if np.linalg.norm(ax) < 1e-6:
            ax = np.cross(u, [0, 1, 0])
        ax /= np.linalg.norm(ax)
        return np.array([0.0, *ax])
    q = np.array([1.0 + c, *np.cross(u, v)])
    return q / np.linalg.norm(q)


def start_value(info):
    """The controller's start pose: fingers/thumb open, wrist and lateral axes centred."""
    return info["open"] if info["role"] == "flex" else info["neutral"]


def open_pose_data(m, joints):
    d = mujoco.MjData(m)
    for n, info in joints.items():
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        if j >= 0:
            d.qpos[m.jnt_qposadr[j]] = start_value(info)
    mujoco.mj_forward(m, d)
    return d


def lowest_robot_z(m, d, root_id):
    """Lowest point of every geom in the robot subtree (from each geom's box)."""
    zmin = np.inf
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        while b not in (0, root_id):
            b = m.body_parentid[b]
        if b != root_id:
            continue
        c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
        R = d.geom_xmat[g].reshape(3, 3)
        pts = d.geom_xpos[g] + (c + signs * h) @ R.T
        zmin = min(zmin, float(pts[:, 2].min()))
    return zmin


def point_up(tree, root, out, m, joints, palm_name):
    palm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, palm_name)
    b = palm
    while m.body_parentid[b] != 0:
        b = m.body_parentid[b]
    root_id = b
    name = m.body(root_id).name
    el = next((e for e in root.find("worldbody").iter("body") if e.get("name") == name), None)
    if el is None or not name:
        print("[point-up] could not find the robot's root body - skipped")
        return m
    # the forearm's own long axis = its minimum-inertia principal axis (a cylinder's centre line),
    # pointed toward the hand. Fallback: forearm centre of mass -> hand centre of mass.
    d = open_pose_data(m, joints)
    to_hand = d.subtree_com[palm] - d.xipos[root_id]
    I = np.array(m.body_inertia[root_id])
    order = np.argsort(I)
    if I[order[0]] < 0.8 * I[order[1]]:
        direction = np.array(d.ximat[root_id]).reshape(3, 3)[:, order[0]]
        if direction @ to_hand < 0:
            direction = -direction
        how = "forearm long axis"
    else:
        direction = to_hand
        how = "forearm -> hand centre of mass"
    if np.linalg.norm(direction) < 1e-6:
        print("[point-up] could not find an up direction - skipped")
        return m
    q_add = quat_from_to(direction, np.array([0.0, 0.0, 1.0]))
    q_new = quat_mul(q_add, m.body_quat[root_id])
    for a in ("quat", "euler", "axisangle", "xyaxes", "zaxis"):
        el.attrib.pop(a, None)
    el.set("quat", fmt(*q_new))
    el.set("pos", fmt(*m.body_pos[root_id]))
    tree.write(out)
    m = mujoco.MjModel.from_xml_path(out)

    # rest the lowest point of the arm 1 mm above the floor (z = 0)
    d = open_pose_data(m, joints)
    zmin = lowest_robot_z(m, d, root_id)
    pos = np.array(m.body_pos[root_id]) + [0.0, 0.0, 0.001 - zmin]
    el.set("pos", fmt(*pos))
    tree.write(out)
    m = mujoco.MjModel.from_xml_path(out)
    d = open_pose_data(m, joints)
    R = np.array(d.ximat[root_id]).reshape(3, 3)
    now = R[:, order[0]] if how == "forearm long axis" else d.subtree_com[palm] - d.xipos[root_id]
    now = now * np.sign(now[2] or 1.0)
    tilt = math.degrees(math.acos(min(1.0, abs(now[2]) / np.linalg.norm(now))))
    print(f"[point-up] {how} now {np.round(now / np.linalg.norm(now), 3).tolist()} "
          f"(tilt from vertical {tilt:.2f} deg; was {np.round(direction / np.linalg.norm(direction), 3).tolist()}); "
          f"arm base at z = {lowest_robot_z(m, d, root_id) * 1000:.1f} mm")
    return m


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mjcf", help="MJCF (.xml) written by sw2robot --mujoco")
    ap.add_argument("-o", "--out", help="output xml (default: <input>_actuated.xml)")
    ap.add_argument("--config", help="controller config path (default: hand_config.json next to output)")
    ap.add_argument("--weld-base", action="store_true", help="remove the free joint so the forearm is fixed to the world")
    ap.add_argument("--add-axis", action="append", default=[], metavar="NAME[:x,y,z][@lo,hi]",
                    help="add a 2nd hinge in the same body as the joint NAME (a joint name, or part of a "
                         "joint/body name, e.g. Thumb_Base_1). No axis = auto (perpendicular to the existing "
                         "axis and to the bone). @lo,hi = range in degrees. Repeatable.")
    ap.add_argument("--pivot-to-parent-center", action="append", default=[], metavar="BODY",
                    help="move the joints of BODY (part of a body name, e.g. Palm_1) onto the centre of mass "
                         "of its parent body - for a ball wrist, so the palm rotates about the ball centre")
    ap.add_argument("--add-axis-range", nargs=2, type=float, default=[-20.0, 20.0], metavar=("LO", "HI"),
                    help="range for added axes in degrees (default -20 20)")
    ap.add_argument("--kp", type=float, help="fixed kp for every joint [N*m/rad]; default = auto per joint")
    ap.add_argument("--sag", type=float, default=0.02, help="auto kp: allowed droop under gravity [rad] (default 0.02)")
    ap.add_argument("--kp-min", type=float, default=0.02, help="lower bound for auto kp")
    ap.add_argument("--zeta", type=float, default=1.0, help="damping ratio (1 = critical)")
    ap.add_argument("--force-margin", type=float, default=0.5,
                    help="force limit = kp * this, i.e. error [rad] at which the actuator saturates")
    ap.add_argument("--no-anatomical", action="store_true",
                    help="keep the exported joint zero/axis sign (default: 0 = straight finger, + = flexion)")
    ap.add_argument("--flex-range", nargs=2, type=float, default=[0.0, 90.0], metavar=("LO", "HI"),
                    help="finger/thumb flexion range in degrees, 0 = straight (default 0 90)")
    ap.add_argument("--straight-tol", type=float, default=10.0, metavar="DEG",
                    help="a joint bent less than this in CAD is taken as straight as-is (default 10)")
    ap.add_argument("--cmc-range", nargs=2, type=float, default=[0.0, 90.0], metavar=("LO", "HI"),
                    help="thumb CMC range in degrees, 0 = thumb in the palm plane (default 0 90)")
    ap.add_argument("--fix-wrist", action="store_true",
                    help="rebuild the 2 wrist axes from the hand: flexion parallel to the knuckles, "
                         "tilt perpendicular to the palm")
    ap.add_argument("--point-up", action="store_true",
                    help="rotate the whole model so forearm -> palm points up (+Z) and rest it on the floor")
    ap.add_argument("--keep-contacts", action="store_true",
                    help="do NOT auto-exclude body pairs that touch in the open pose")
    args = ap.parse_args()

    src = os.path.abspath(args.mjcf)
    workdir = os.path.dirname(src)
    out = os.path.abspath(args.out) if args.out else os.path.splitext(src)[0] + "_actuated.xml"
    cfg_path = os.path.abspath(args.config) if args.config else os.path.join(os.path.dirname(out), "hand_config.json")

    tree = ET.parse(src)
    root = tree.getroot()
    comp = root.find("compiler")
    angle_unit = comp.get("angle", "degree") if comp is not None else "degree"
    ang = (lambda r: math.degrees(r)) if angle_unit == "degree" else (lambda r: r)
    structure_changed = False

    # 1. every joint needs a name (the controller works by name)
    for i, j in enumerate(world_joints(root)):
        if not j.get("name") and j.get("type", "hinge") != "free":
            j.set("name", f"joint_{i}")

    # 2. optional: weld the base
    if args.weld_base:
        for body in root.iter("body"):
            for ch in list(body):
                if ch.tag == "freejoint" or (ch.tag == "joint" and ch.get("type") == "free"):
                    body.remove(ch)
                    structure_changed = True
                    drop_keyframes(root)
                    print(f"[base] removed free joint from body '{body.get('name')}'")

    m0 = compile_tree(tree, workdir)
    if any(int(t) == FREE for t in m0.jnt_type):
        print("[warn] the model still has a FREE joint - the arm will fall. Re-export with "
              "--mujoco-fixed-base or rerun this script with --weld-base.")

    # 3. optional: extra hinge axes
    extra_names = set()
    lo_add, hi_add = (math.radians(v) for v in args.add_axis_range)
    pmap = parent_map(root)
    for spec in args.add_axis:
        rng = None
        if "@" in spec:
            spec, _, rtxt = spec.partition("@")
            rng = [math.radians(float(v)) for v in rtxt.split(",")]
        key, _, axis_txt = spec.partition(":")
        jid = resolve_joint(m0, key)
        jname = m0.joint(jid).name
        el = find_joint_elem(root, jname)
        if el is None:
            sys.exit(f"[add-axis] joint '{jname}' not found in the XML")
        a1 = np.array(m0.jnt_axis[jid], float)
        if axis_txt:
            a2 = np.array([float(v) for v in axis_txt.split(",")])
        else:  # auto: perpendicular to the existing axis and to the bone (joint -> body COM)
            bone = np.array(m0.body_ipos[m0.jnt_bodyid[jid]]) - np.array(m0.jnt_pos[jid])
            a2 = np.cross(a1, bone)
            if np.linalg.norm(a2) < 1e-9:
                sys.exit(f"[add-axis] can't infer an axis for '{jname}'; give one: {key}:x,y,z")
        a2 = a2 / np.linalg.norm(a2)
        lo, hi = rng if rng else (lo_add, hi_add)
        new = ET.Element("joint", name=f"{jname}_abd", type="hinge",
                         pos=fmt(*m0.jnt_pos[jid]), axis=fmt(*a2),
                         range=fmt(ang(lo), ang(hi)), limited="true")
        body_el = pmap[el]
        body_el.insert(list(body_el).index(el) + 1, new)
        extra_names.add(f"{jname}_abd")
        structure_changed = True
        drop_keyframes(root)
        print(f"[add-axis] {jname}_abd  axis(body frame) = {fmt(*a2)}  "
              f"range [{math.degrees(lo):.0f}, {math.degrees(hi):.0f}] deg")

    # 3b. optional: move a body's joints onto the centre of its parent (ball-type wrist)
    for key in args.pivot_to_parent_center:
        mt = compile_tree(tree, workdir)
        dt = mujoco.MjData(mt)
        mujoco.mj_forward(mt, dt)
        b = resolve_body(mt, key)
        par = int(mt.body_parentid[b])
        centre = dt.xipos[par]                                  # parent's centre of mass
        R = dt.xmat[b].reshape(3, 3)
        local = R.T @ (centre - dt.xpos[b])
        body_el = next(e for e in root.find("worldbody").iter("body") if e.get("name") == mt.body(b).name)
        moved = [c for c in body_el if c.tag == "joint"]
        for c in moved:
            old_pos = c.get("pos", "0 0 0")
            c.set("pos", fmt(*local))
            print(f"[pivot] {c.get('name')}: pos {old_pos} -> {fmt(*local)} "
                  f"(centre of '{mt.body(par).name}')")
        if not moved:
            print(f"[pivot] body '{mt.body(b).name}' has no joints - nothing moved")

    m1 = compile_tree(tree, workdir)
    palm, joints, fingers, wrist, spread, skipped, wrist_axes = analyse(m1, extra_names)
    if not args.no_anatomical or args.fix_wrist:
        lo_f, hi_f = (math.radians(v) for v in args.flex_range)
        anatomical_setup(root, m1, palm, fingers if not args.no_anatomical else {}, wrist_axes,
                         lo_f, hi_f, args.fix_wrist, ang, *(math.radians(v) for v in args.cmc_range),
                         math.radians(args.straight_tol))
        m1 = compile_tree(tree, workdir)
        palm, joints, fingers, wrist, spread, skipped, wrist_axes = analyse(m1, extra_names)
    if not joints:
        sys.exit("No hinge/slide joints found - check the joint types in the sw2robot editor (key 't').")

    # 4. ranges, open / closed poses
    for n, info in joints.items():
        j = info["id"]
        if m1.jnt_limited[j]:
            lo, hi = (float(v) for v in m1.jnt_range[j])
            info["range_added"] = False
        else:
            if info["type"] == SLIDE:
                lo, hi = -0.01, 0.01
            elif info["role"] == "wrist":
                lo, hi = -math.pi / 4, math.pi / 4
            else:
                lo, hi = -math.pi / 2, math.pi / 2
            info["range_added"] = True
            print(f"[warn] '{n}' had no limits - set to [{math.degrees(lo):.0f}, {math.degrees(hi):.0f}] deg. "
                  f"Set real limits in SolidWorks / the sw2robot editor.")
        neutral = min(max(0.0, lo), hi)
        closed = hi if abs(hi - neutral) >= abs(lo - neutral) else lo
        opened = lo if closed == hi else hi          # the OTHER limit -> trigger covers the full range
        info.update(range=[lo, hi], neutral=neutral, open=opened, closed=closed)
        if info["role"] == "flex" and abs(abs(hi - neutral) - abs(lo - neutral)) < 1e-3 and hi - lo > 1e-6:
            print(f"[warn] '{n}' range is symmetric about 0 - guessed +limit as 'closed'. "
                  f"Check with --sweep and edit 'closed' in hand_config.json if it bends backwards.")

    # 5. gains from the load each joint carries, evaluated in the open pose
    d1 = mujoco.MjData(m1)
    for n, info in joints.items():
        d1.qpos[m1.jnt_qposadr[info["id"]]] = start_value(info)
    mujoco.mj_forward(m1, d1)
    g = float(np.linalg.norm(m1.opt.gravity)) or 9.81
    for n, info in joints.items():
        j, b = info["id"], info["body"]
        msub = float(m1.body_subtreemass[b])
        if info["type"] == HINGE:
            r = np.array(d1.subtree_com[b]) - np.array(d1.xanchor[j])
            ax = np.array(d1.xaxis[j])
            lever = float(np.linalg.norm(r - ax * r.dot(ax)))
            tau_g = msub * g * lever
            inertia = msub * lever ** 2
        else:
            tau_g, inertia = msub * g, msub
        inertia = max(inertia, 1e-8)
        kp = args.kp if args.kp else max(args.kp_min, tau_g / args.sag)
        armature = inertia                                   # doubles effective inertia -> stable at 2 ms
        damping = 2 * args.zeta * math.sqrt(kp * (inertia + armature))
        force = kp * args.force_margin
        info.update(kp=kp, damping=damping, armature=armature, force=force, tau_g=tau_g)

    # 6. write joint dynamics / ranges
    for n, info in joints.items():
        el = find_joint_elem(root, n)
        el.set("damping", fmt(info["damping"]))
        el.set("armature", fmt(info["armature"]))
        if info["range_added"]:
            el.set("range", fmt(ang(info["range"][0]), ang(info["range"][1])))
            el.set("limited", "true")

    # 7. actuators (replace sw2robot's)
    for a in root.findall("actuator"):
        root.remove(a)
    act = ET.SubElement(root, "actuator")
    for n, info in joints.items():
        info["actuator"] = f"act_{n}"
        ET.SubElement(act, "position", name=info["actuator"], joint=n,
                      kp=fmt(info["kp"]), ctrllimited="true", ctrlrange=fmt(*info["range"]),
                      forcelimited="true", forcerange=fmt(-info["force"], info["force"]))

    # 8. joint position sensors
    sens = get_or_make(root, "sensor")
    existing = {s.get("name") for s in sens}
    for n in joints:
        if f"q_{n}" not in existing:
            ET.SubElement(sens, "jointpos", name=f"q_{n}", joint=n)

    # 9. keyframes: ctrl size changed; qpos size changed if we added / removed joints
    for kf in root.findall("keyframe"):
        if structure_changed:
            root.remove(kf)
        else:
            for key in kf:
                for attr in ("ctrl", "act"):
                    key.attrib.pop(attr, None)

    # 10. solver: implicit damping handles the small finger inertias well
    opt = root.find("option")
    if opt is None:
        opt = ET.Element("option")
        root.insert(1 if comp is not None else 0, opt)
    opt.set("integrator", opt.get("integrator", "implicitfast"))

    tree.write(out)
    m2 = mujoco.MjModel.from_xml_path(out)

    # 10b. optional: stand the arm up (forearm -> palm along +Z), resting on the floor
    if args.point_up:
        m2 = point_up(tree, root, out, m2, joints, m1.body(palm).name)

    # 11. exclude pairs already in contact at the open pose (CAD clearances are often ~0)
    excluded = []
    if not args.keep_contacts:
        d2 = mujoco.MjData(m2)
        for n, info in joints.items():
            d2.qpos[m2.jnt_qposadr[mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_JOINT, n)]] = start_value(info)
        mujoco.mj_forward(m2, d2)
        pairs = set()
        for i in range(d2.ncon):
            c = d2.contact[i]
            b1, b2 = int(m2.geom_bodyid[c.geom1]), int(m2.geom_bodyid[c.geom2])
            n1, n2 = m2.body(b1).name, m2.body(b2).name
            if b1 and b2 and b1 != b2 and n1 and n2:
                pairs.add(tuple(sorted((n1, n2))))
        if pairs:
            cont = get_or_make(root, "contact")
            for n1, n2 in sorted(pairs):
                ET.SubElement(cont, "exclude", body1=n1, body2=n2)
                excluded.append(f"{n1} <-> {n2}")
            tree.write(out)
            m2 = mujoco.MjModel.from_xml_path(out)

    # 12. controller config
    cfg = dict(
        model=os.path.relpath(out, os.path.dirname(cfg_path)),
        palm_body=m1.body(palm).name,
        wrist=wrist,
        wrist_axes=wrist_axes,
        fingers=fingers,
        spread=spread,
        stage_weights=[1.0, 1.0, 1.0],
        joints={n: dict(actuator=i["actuator"], group=i["group"], role=i["role"],
                        range=[round(v, 5) for v in i["range"]],
                        open=round(i["open"], 5), closed=round(i["closed"], 5),
                        neutral=round(i["neutral"], 5), sign=1,
                        kp=round(i["kp"], 6), force=round(i["force"], 6))
                for n, i in joints.items()},
    )
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    # 13. report
    print(f"\npalm body : {m1.body(palm).name}")
    print(f"wrist     : {wrist_axes or '-- none found --'}")
    for gname, lst in fingers.items():
        print(f"{gname:<10}: {lst}")
    if spread:
        print(f"spread    : {spread}")
    for n, why in skipped:
        print(f"[skip] {n} ({why} joint - not actuated)")
    print(f"\n{'joint':<34}{'kp':>10}{'damping':>11}{'force':>10}  range(deg)")
    for n, i in joints.items():
        print(f"{n:<34}{i['kp']:>10.4g}{i['damping']:>11.3g}{i['force']:>10.3g}  "
              f"[{math.degrees(i['range'][0]):.0f}, {math.degrees(i['range'][1]):.0f}]")
    if excluded:
        print(f"\nexcluded {len(excluded)} contact pair(s) touching in the open pose:")
        for e in excluded:
            print("   ", e)
    print(f"\n{m2.nu} actuators  (expected for your hand after the CAD fixes: 17"
          f"{f' + {len(extra_names)} added axes' if extra_names else ''})")
    print(f"wrote {out}\nwrote {cfg_path}")


if __name__ == "__main__":
    main()
