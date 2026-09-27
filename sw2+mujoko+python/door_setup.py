#!/usr/bin/env python3
"""
door_setup.py
=============
Turns the MJCF that sw2robot exports for Full_door_v4 into a simulation-ready door:

  * upright: the door hinge axis (vertical in SolidWorks = Y) becomes world +Z, base resting on z = 0
  * masses: --mass door=38.4 (kg) rescales mass AND inertia of that part (parts have no material in CAD)
  * passive door: sw2robot's actuators are removed; hinge gets damping + friction
  * handle: return spring (it springs back to its CAD pose) + damping
  * latch: the door is locked closed until the handle is turned more than --unlock-deg
    (an equality constraint that the runtime switches on/off; see --view)
  * contacts that already touch in the CAD pose (door in its frame) are excluded
  * jointpos sensors door_angle / handle_angle, and door_config.json for later scenes

Usage
  python door_setup.py <Full_door_v4.xml> --mass door=38.4 --mass handle=0.5
  python door_setup.py <Full_door_v4.xml> --mass door=38.4 --view        # also open the viewer to try it
  python door_setup.py <Full_door_v4_sim.xml> --view-only                 # just view an existing sim file
  python door_setup.py <Full_door_v4_sim.xml> --check                     # measure hinge + handle joints
  python door_setup.py <Full_door_v4_sim.xml> --sweep                     # watch both joints move

In the viewer: double-click the handle, then Ctrl + right-drag to turn it; double-click the door and
Ctrl + right-drag to push it. The terminal prints LOCKED / UNLOCKED as the latch changes.
"""
import argparse
import json
import math
import os
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)


def fmt(*v):
    return " ".join(f"{float(x):.6g}" for x in v)


def compile_tree(tree, workdir):
    tmp = os.path.join(workdir, "__door_setup_tmp.xml")
    tree.write(tmp)
    try:
        return mujoco.MjModel.from_xml_path(tmp)
    finally:
        os.remove(tmp)


def find(root, tag, name):
    wb = root.find("worldbody")
    return next((e for e in wb.iter(tag) if e.get("name") == name), None)


def body_by_key(m, key):
    hits = [b for b in range(1, m.nbody) if key.lower() in m.body(b).name.lower()]
    exact = [b for b in hits if m.body(b).name.lower().split("_")[0] == key.lower()]
    hits = exact or hits
    if len(hits) != 1:
        sys.exit(f"'{key}' matches {len(hits)} bodies: {[m.body(b).name for b in hits]}")
    return hits[0]


def quat_from_to(u, v):
    u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
    c = float(u @ v)
    if c < -0.999999:
        ax = np.cross(u, [1, 0, 0])
        if np.linalg.norm(ax) < 1e-6:
            ax = np.cross(u, [0, 1, 0])
        return np.array([0.0, *(ax / np.linalg.norm(ax))])
    q = np.array([1.0 + c, *np.cross(u, v)])
    return q / np.linalg.norm(q)


def quat_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


def rot(q, v):
    out = np.zeros(3)
    mujoco.mju_rotVecQuat(out, np.asarray(v, float), np.asarray(q, float))
    return out


def z_extent(m, d, bodies):
    signs = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
    lo, hi = np.inf, -np.inf
    for g in range(m.ngeom):
        if m.geom_bodyid[g] in bodies:
            c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
            pts = d.geom_xpos[g] + (c + signs * h) @ d.geom_xmat[g].reshape(3, 3).T
            lo, hi = min(lo, float(pts[:, 2].min())), max(hi, float(pts[:, 2].max()))
    return lo, hi


def lowest_z(m, d, bodies):
    return z_extent(m, d, bodies)[0]


# --------------------------------------------------------------------------- latch runtime
class Latch:
    """Door locked at its closed pose until the handle is turned past unlock (rad).
    It re-locks only once the door is back near closed AND the handle is released."""
    def __init__(self, m, cfg):
        self.m = m
        self.eq = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, cfg["latch"]) if cfg.get("latch") else -1
        jh = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, cfg["hinge"])
        jd = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, cfg["handle"])
        self.qh, self.qd = m.jnt_qposadr[jh], m.jnt_qposadr[jd]
        self.h0, self.d0 = float(m.qpos0[self.qh]), float(m.qpos0[self.qd])
        self.unlock = math.radians(cfg.get("unlock_deg", 30))
        self.locked = None

    def update(self, d, verbose=True):
        if self.eq < 0:
            return
        handle_turned = abs(d.qpos[self.qd] - self.d0) > self.unlock
        door_closed = abs(d.qpos[self.qh] - self.h0) < math.radians(2)
        locked = self.locked if self.locked is not None else True
        if locked and handle_turned:
            locked = False
        elif not locked and door_closed and not handle_turned:
            locked = True
        if locked != self.locked:
            d.eq_active[self.eq] = 1 if locked else 0
            if verbose:
                print("[latch] LOCKED" if locked else "[latch] UNLOCKED - door can swing")
        self.locked = locked


def view(path, cfg_path):
    import mujoco.viewer
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    latch = Latch(m, cfg) if cfg else None
    print("[view] double-click the handle, Ctrl + right-drag to turn it; then push the door the same way")
    with mujoco.viewer.launch_passive(m, d) as v:
        wall0, sim0, last_print = time.perf_counter(), d.time, 0.0
        while v.is_running():
            now = time.perf_counter()
            if latch:
                with v.lock():
                    latch.update(d)
            while d.time - sim0 < now - wall0:
                mujoco.mj_step(m, d)
            if latch and now - last_print > 1.0:
                last_print = now
                print(f"   door {math.degrees(d.qpos[latch.qh] - latch.h0):6.1f} deg   "
                      f"handle {math.degrees(d.qpos[latch.qd] - latch.d0):6.1f} deg")
            v.sync()
            time.sleep(max(0.0, 1 / 90 - (time.perf_counter() - now)))


# --------------------------------------------------------------------------- frame collision boxes
MESH = int(mujoco.mjtGeom.mjGEOM_MESH)


def geom_points(m, d, g):
    """World points of a geom: mesh vertices, or the corners of its bounding box."""
    R = np.array(d.geom_xmat[g]).reshape(3, 3)
    if int(m.geom_type[g]) == MESH and m.geom_dataid[g] >= 0:
        mid = m.geom_dataid[g]
        v = np.array(m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]])
    else:
        c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
        v = np.array([[a, b, cc] for a in (-1, 1) for b in (-1, 1) for cc in (-1, 1)]) * h + c
    return np.array(d.geom_xpos[g]) + v @ R.T


def hull_faces(m, mid):
    """Triangles (vertex indices) of the convex hull MuJoCo uses to collide mesh `mid`, or None."""
    try:
        adr = int(m.mesh_graphadr[mid])
        if adr < 0:
            return None
        gr = m.mesh_graph
        nv, nf = int(gr[adr]), int(gr[adr + 1])
        off = adr + 2 + nv + nv + (nv + 3 * nf - 6)          # skip vert_edgeadr, vert_globalid, edge_localid
        return np.array(gr[off: off + 3 * nf]).reshape(nf, 3)
    except Exception:
        return None


def geom_points_dense(m, d, g, n=8):
    """World points densely covering a geom's surface: points along every mesh triangle edge, or a grid on
    the surface of a primitive's box. (Vertices alone miss the middle of long parts such as rods.)"""
    R = np.array(d.geom_xmat[g]).reshape(3, 3)
    t = np.linspace(0.0, 1.0, n)[:, None]
    if int(m.geom_type[g]) == MESH and m.geom_dataid[g] >= 0:
        mid = m.geom_dataid[g]
        V = np.array(m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]])
        F = hull_faces(m, mid)                       # MuJoCo collides meshes as their CONVEX HULL
        if F is None:
            F = np.array(m.mesh_face[m.mesh_faceadr[mid]: m.mesh_faceadr[mid] + m.mesh_facenum[mid]])
        pts = [V]
        if len(F):
            A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
            for P, Q in ((A, B), (B, C), (C, A)):    # points along every edge
                pts.append((P[None] + t[:, :, None] * (Q - P)[None]).reshape(-1, 3))
            for w1 in np.linspace(0.1, 0.8, 5):      # and inside every (hull) face
                for w2 in np.linspace(0.1, 0.9 - w1, 4):
                    pts.append(A * (1 - w1 - w2) + B * w1 + C * w2)
        v = np.unique(np.round(np.vstack(pts), 5), axis=0)
    else:
        c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
        g1 = np.linspace(-1, 1, n)
        grid = np.array(np.meshgrid(g1, g1, g1)).reshape(3, -1).T
        grid = grid[np.any(np.isclose(np.abs(grid), 1.0), axis=1)]          # surface only
        v = c + grid * h
    return np.array(d.geom_xpos[g]) + v @ R.T


def _geom_label(m, g):
    lab = m.geom(g).name or ""
    if int(m.geom_type[g]) == MESH and m.geom_dataid[g] >= 0:
        lab += " " + m.mesh(m.geom_dataid[g]).name
    return lab.lower()


def frame_boxes(root, m, d, hinge, handle, margin=0.002, pocket=False):
    """Replace the frame's convex-hull collision with boxes (jambs, header, sill) and cut a pocket
    in the latch-side jamb where the handle's protruding end sits and swings. Returns a report."""
    jh, jd = m.joint(hinge).id, m.joint(handle).id
    door_b, hand_b = m.jnt_bodyid[jh], m.jnt_bodyid[jd]
    fb = m.body_parentid[door_b]
    coll = lambda g: bool(m.geom_contype[g] or m.geom_conaffinity[g])
    fgeoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == fb and coll(g) and "frame" in _geom_label(m, g)]
    if not fgeoms:                                   # no 'frame' in the names: take the tall collision geoms
        fgeoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == fb and coll(g)
                  and np.ptp(geom_points(m, d, g)[:, 2]) > 0.5]
    if not fgeoms:
        return "[frame-boxes] could not find the frame's collision geoms - skipped"

    # door-aligned frame: z = hinge axis (up), u = hinge -> door centre, v = door normal
    p = np.array(d.xanchor[jh])
    z = np.array(d.xaxis[jh]) * (1 if d.xaxis[jh][2] >= 0 else -1)
    u = np.array(d.xipos[door_b]) - p
    u -= z * (u @ z)
    u /= np.linalg.norm(u)
    v = np.cross(z, u)
    uvz = lambda P: np.stack([(P - p) @ u, (P - p) @ v, (P - p) @ z], -1)
    F = uvz(np.vstack([geom_points(m, d, g) for g in fgeoms]))
    dg = [g for g in range(m.ngeom) if m.geom_bodyid[g] == door_b]
    D = uvz(np.vstack([geom_points(m, d, g) for g in dg]))
    (fu0, fv0, fz0), (fu1, fv1, fz1) = F.min(0), F.max(0)
    (du0, dv0, dz0), (du1, dv1, dz1) = D.min(0), D.max(0)

    eps = 1e-4
    # real inner faces of the frame (the frame may stand off the door edge by a gap)
    lat_in = F[F[:, 0] > du1 - eps][:, 0].min() if np.any(F[:, 0] > du1 - eps) else du1
    hin_in = F[F[:, 0] < du0 + eps][:, 0].max() if np.any(F[:, 0] < du0 + eps) else du0
    top_in = F[F[:, 2] > dz1 - eps][:, 2].min() if np.any(F[:, 2] > dz1 - eps) else dz1
    boxes = []                                       # (name, [u0,u1], [v0,v1], [z0,z1])
    if fz1 - top_in > 0.002:
        boxes.append(("collide_header", (hin_in, lat_in), (fv0, fv1), (top_in, fz1)))
    if dz0 - fz0 > 0.02:                             # a real sill, not just the gap under the door
        boxes.append(("collide_sill", (du0, du1), (fv0, fv1), (fz0, dz0)))

    # handle points (whole handle range + margin) that reach into EITHER jamb
    adr = m.jnt_qposadr[jd]
    hg = [g for g in range(m.ngeom) if m.geom_bodyid[g] == hand_b and coll(g)] or \
         [g for g in range(m.ngeom) if m.geom_bodyid[g] == hand_b]
    allpts = []
    lo_h, hi_h = m.jnt_range[jd] if m.jnt_limited[jd] else (-math.pi, math.pi)
    angles = list(np.linspace(lo_h - math.radians(15), hi_h + math.radians(5), 111))
    for q in angles:
        d.qpos[:] = m.qpos0
        d.qpos[adr] = q
        mujoco.mj_forward(m, d)
        allpts.append(uvz(np.vstack([geom_points_dense(m, d, g) for g in hg])))
    d.qpos[:] = m.qpos0
    mujoco.mj_forward(m, d)
    allpts = np.vstack(allpts)

    in_depth = (allpts[:, 1] > fv0) & (allpts[:, 1] < fv1) & (allpts[:, 2] > fz0) & (allpts[:, 2] < fz1)

    # where does the CAD handle intersect the real frame? (at rest, and while turning with the door closed)
    n_ang = len(angles)
    per = len(allpts) // n_ang
    q_rest = float(m.qpos0[adr])
    in_frame = ((allpts[:, 0] < hin_in) | (allpts[:, 0] > lat_in) | (allpts[:, 2] > top_in)) & in_depth
    hit_ang = [math.degrees(angles[k] - q_rest) for k in range(n_ang)
               if lo_h - 1e-6 <= angles[k] <= hi_h + 1e-6 and in_frame[k * per:(k + 1) * per].any()]
    k_rest = int(np.argmin(np.abs(np.array(angles) - q_rest)))
    diag = []
    if in_frame[k_rest * per:(k_rest + 1) * per].any():
        diag.append("the CAD handle OVERLAPS the frame with the door closed (at rest)")
    if hit_ang:
        diag.append(f"turning the handle with the door closed hits the frame between "
                    f"{min(hit_ang):.0f} and {max(hit_ang):.0f} deg")

    notes = []
    if not pocket:
        for side, (j0, j1) in (("hinge", (fu0, hin_in)), ("latch", (lat_in, fu1))):
            if j1 - j0 > 0.002:
                boxes.append((f"collide_jamb_{side}", (j0, j1), (fv0, fv1), (fz0, fz1)))
        note = "SOLID frame, exactly the real shape (no pockets)"
        if diag:
            note += ("; WARNING: " + "; ".join(diag) + " -> with a solid frame the handle cannot move there: "
                     "change the handle or frame in CAD (or use --pocket)")
        else:
            note += "; the handle is clear of the frame at every handle angle"
    else:
        for side, (j0, j1), inside in (("hinge", (fu0, hin_in), (allpts[:, 0] < hin_in) & in_depth),
                                       ("latch", (lat_in, fu1), (allpts[:, 0] > lat_in) & in_depth)):
            if j1 - j0 <= 0.002:
                continue
            pts = allpts[inside]
            tag = f"collide_jamb_{side}"
            if not len(pts):
                boxes.append((tag, (j0, j1), (fv0, fv1), (fz0, fz1)))
                continue
            pv0, pv1 = max(pts[:, 1].min() - margin, fv0), min(pts[:, 1].max() + margin, fv1)
            pz0, pz1 = max(pts[:, 2].min() - margin, fz0), min(pts[:, 2].max() + margin, fz1)
            if side == "latch":
                pe = min(pts[:, 0].max() + margin, fu1)
                depth, end = pe - lat_in, (pe, fu1)
            else:
                pe = max(pts[:, 0].min() - margin, fu0)
                depth, end = hin_in - pe, (fu0, pe)
            boxes += [(tag + "_below", (j0, j1), (fv0, fv1), (fz0, pz0)),
                      (tag + "_above", (j0, j1), (fv0, fv1), (pz1, fz1)),
                      (tag + "_front", (j0, j1), (pv1, fv1), (pz0, pz1)),
                      (tag + "_back", (j0, j1), (fv0, pv0), (pz0, pz1)),
                      (tag + "_end", end, (pv0, pv1), (pz0, pz1))]
            notes.append(f"pocket cut in the {side}-side jamb ({depth * 1000:.0f} mm deep, "
                         f"{(pz1 - pz0) * 1000:.0f} mm tall, {(pv1 - pv0) * 1000:.0f} mm across)")
        note = "; ".join(notes) if notes else "no pockets needed - the handle stays clear of the frame"

    # the handle must be able to collide: if none of its geoms does, switch collision on for all of them
    hel = find(root, "body", m.body(hand_b).name)
    hxml = [c for c in hel if c.tag == "geom"]
    if not any(coll(g) for g in range(m.ngeom) if m.geom_bodyid[g] == hand_b):
        for c in hxml:
            c.set("contype", "1")
            c.set("conaffinity", "1")
        note_h = " handle collision switched ON (it had none);"
    else:
        note_h = ""

    # write: boxes into the frame body, original frame collision switched off
    fname = m.body(fb).name
    fel = find(root, "body", fname)
    for c in [c for c in fel if c.tag == "geom" and (c.get("name") or "").startswith("collide_")]:
        fel.remove(c)
    geoms_xml = [c for c in fel if c.tag == "geom"]
    first = m.body_geomadr[fb]
    for g in fgeoms:
        k = g - first
        if 0 <= k < len(geoms_xml):
            geoms_xml[k].set("contype", "0")
            geoms_xml[k].set("conaffinity", "0")
    Rb = np.array(d.xmat[fb]).reshape(3, 3)
    B = np.stack([u, v, z], 1)
    qloc = np.zeros(4)
    mujoco.mju_mat2Quat(qloc, (Rb.T @ B).flatten())
    made = []
    for name, (a0, a1), (b0, b1), (c0, c1) in boxes:
        half = np.array([a1 - a0, b1 - b0, c1 - c0]) / 2
        if np.any(half < 0.00025):
            continue
        c = p + u * (a0 + a1) / 2 + v * (b0 + b1) / 2 + z * (c0 + c1) / 2
        ET.SubElement(fel, "geom", name=name, type="box", size=fmt(*half),
                      pos=fmt(*(Rb.T @ (c - np.array(d.xpos[fb])))), quat=fmt(*qloc),
                      contype="1", conaffinity="1", group="3", rgba="0.9 0.2 0.2 0.35")
        made.append(name)
    return f"[frame-boxes] frame collision = {len(made)} boxes ({', '.join(made)});{note_h} {note}"


# --------------------------------------------------------------------------- joint checks
def _load_sim(path, cfg_path):
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    cfg = json.load(open(cfg_path))
    jh = m.joint(cfg["hinge"]).id
    jd = m.joint(cfg["handle"]).id
    return m, d, cfg, jh, jd


def _contacts(m, d, skip_world=True):
    out = []
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        if skip_world and (b1 == 0 or b2 == 0):
            continue
        out.append((m.body(b1).name, m.body(b2).name, float(c.dist)))
    return out


def check(path, cfg_path):
    """Measure both joints and sweep them kinematically, reporting collisions."""
    m, d, cfg, jh, jd = _load_sim(path, cfg_path)
    mujoco.mj_forward(m, d)
    mm = lambda v: "[" + ", ".join(f"{x * 1000:7.1f}" for x in v) + "] mm"
    deg = math.degrees
    db, hb = m.jnt_bodyid[jh], m.jnt_bodyid[jd]

    def line_dist(p, a, x):                        # distance of point x from the line (p, a)
        r = np.asarray(x) - p
        return float(np.linalg.norm(r - a * (r @ a)))

    ah, ph = np.array(d.xaxis[jh]), np.array(d.xanchor[jh])
    ad, pd = np.array(d.xaxis[jd]), np.array(d.xanchor[jd])
    I = np.array(m.body_inertia[db])
    door_normal = np.array(d.ximat[db]).reshape(3, 3)[:, int(np.argmax(I))]   # thin slab: max-inertia axis
    print("\nDOOR HINGE  (" + cfg["hinge"] + ")")
    print(f"   axis            {np.round(ah, 3).tolist()}   -> {deg(math.acos(min(1, abs(ah[2])))):.1f} deg from vertical (want ~0)")
    print(f"   pivot           {mm(ph)}")
    print(f"   range           [{deg(m.jnt_range[jh][0]):.0f}, {deg(m.jnt_range[jh][1]):.0f}] deg")
    print(f"   door centre     {line_dist(ph, ah, d.xipos[db]) * 1000:.0f} mm from the hinge line (want ~half the door width)")
    print("\nHANDLE  (" + cfg["handle"] + ")")
    print(f"   axis            {np.round(ad, 3).tolist()}   -> {deg(math.acos(min(1, abs(ad[2])))):.1f} deg from vertical (want ~90: horizontal)")
    print(f"                   {deg(math.acos(min(1, abs(float(ad @ door_normal))))):.1f} deg from the door's face normal (want ~0: through the door)")
    print(f"   pivot height    {pd[2] * 1000:.0f} mm above the floor")
    print(f"   pivot           {line_dist(ph, ah, pd) * 1000:.0f} mm from the hinge line (want ~door width minus a margin)")
    print(f"   range           [{deg(m.jnt_range[jd][0]):.0f}, {deg(m.jnt_range[jd][1]):.0f}] deg")
    lever = np.array(d.xipos[hb]) - pd
    lever -= ad * (lever @ ad)
    print(f"   lever points    {np.round(lever / (np.linalg.norm(lever) or 1), 2).tolist()} at rest "
          f"(z ~0 = horizontal lever)")

    # kinematic sweeps: where does the door go, does anything collide?
    base_contacts = {(a, b) for a, b, _ in _contacts(m, d)}
    hq = m.jnt_range[jd][int(np.argmax(np.abs(m.jnt_range[jd])))]
    for name, j, q_lo, q_hi in (("door", jh, *m.jnt_range[jh]), ("door, handle turned", jh, *m.jnt_range[jh]),
                                ("handle", jd, *m.jnt_range[jd])):
        adr = m.jnt_qposadr[j]
        other = {m.jnt_qposadr[jd]: hq} if name.startswith("door, handle") else {}
        q0 = float(m.qpos0[adr])
        worst, hits = 0.0, set()
        for q in np.linspace(q_lo, q_hi, 41):
            d.qpos[:] = m.qpos0
            for k, val in other.items():
                d.qpos[k] = val
            d.qpos[adr] = q
            mujoco.mj_forward(m, d)
            for a, b, dist in _contacts(m, d):
                if (a, b) not in base_contacts and dist < -0.001:
                    worst = min(worst, dist)
                    hits.add(f"{a}<->{b}")
        d.qpos[:] = m.qpos0
        d.qpos[adr] = q_hi
        mujoco.mj_forward(m, d)
        moved = np.array(d.xipos[m.jnt_bodyid[j]])
        d.qpos[adr] = q0
        mujoco.mj_forward(m, d)
        moved -= np.array(d.xipos[m.jnt_bodyid[j]])
        coll = f"collides: {', '.join(sorted(hits))} (up to {-worst * 1000:.0f} mm)" if hits else "no collisions"
        print(f"\nsweep {name:<20} {deg(q_lo):6.0f} -> {deg(q_hi):4.0f} deg: at the upper limit its centre moved "
              f"{mm(moved)};  {coll}")
    print("\nAxes/pivots OK? Then look at it move:  python door_setup.py <sim.xml> --sweep")


def sweep_view(path, cfg_path):
    """Animate the door then the handle through their ranges (kinematic, latch ignored)."""
    import mujoco.viewer
    m, d, cfg, jh, jd = _load_sim(path, cfg_path)
    plan = []
    for name, j in (("door", jh), ("handle", jd)):
        lo, hi = m.jnt_range[j]
        q0 = float(m.qpos0[m.jnt_qposadr[j]])
        plan.append((name, j, [q0, hi, lo, q0]))
    print("[sweep] door: rest -> upper -> lower -> rest, then the handle. Close the window to stop.")
    with mujoco.viewer.launch_passive(m, d) as v:
        t0 = time.perf_counter()
        seg_time, shown = 2.0, None
        while v.is_running():
            t = (time.perf_counter() - t0) % (len(plan) * 3 * seg_time)
            k, rem = divmod(t, 3 * seg_time)
            name, j, pts = plan[int(k)]
            s_i, frac = divmod(rem, seg_time)
            a, b = pts[int(s_i)], pts[int(s_i) + 1]
            q = a + (b - a) * (0.5 - 0.5 * math.cos(math.pi * frac / seg_time))
            if shown != (name, int(s_i)):
                shown = (name, int(s_i))
                print(f"   {name}: {math.degrees(a):6.1f} -> {math.degrees(b):6.1f} deg")
            with v.lock():
                d.qpos[:] = m.qpos0
                d.qpos[m.jnt_qposadr[j]] = q
                mujoco.mj_forward(m, d)
            v.sync()
            time.sleep(1 / 60)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mjcf")
    ap.add_argument("-o", "--out", help="output (default <name>_sim.xml next to the input)")
    ap.add_argument("--mass", action="append", default=[], metavar="PART=KG",
                    help="set a part's mass (scales its inertia too), e.g. door=38.4. Repeatable.")
    ap.add_argument("--hinge", help="door hinge joint (default: the joint that moves the 'door' part)")
    ap.add_argument("--handle", help="handle joint (default: the joint that moves the 'handle' part)")
    ap.add_argument("--door-damping", type=float, default=2.0, help="hinge damping N*m*s/rad (default 2)")
    ap.add_argument("--door-friction", type=float, default=0.5, help="hinge friction N*m (default 0.5)")
    ap.add_argument("--handle-spring", type=float, default=4.0, help="handle return spring N*m/rad (default 4)")
    ap.add_argument("--handle-damping", type=float, default=0.02, help="handle damping (default 0.02)")
    ap.add_argument("--unlock-deg", type=float, default=30.0,
                    help="handle turn needed to unlatch the door (default 30 deg)")
    ap.add_argument("--no-latch", action="store_true", help="no latch: the door always swings freely")
    ap.add_argument("--handle-hits-frame", action="store_true",
                    help="physical latch: frame collision rebuilt as boxes with a pocket for the handle's end, "
                         "so the handle stops the door until it is turned (replaces the constraint latch)")
    ap.add_argument("--pocket", action="store_true",
                    help="with --handle-hits-frame: cut a pocket where the CAD handle passes through the frame "
                         "(default: SOLID frame, exactly the real shape)")
    ap.add_argument("--keep-latch", action="store_true", help="with --handle-hits-frame: keep the constraint latch too")
    ap.add_argument("--keep-actuators", action="store_true", help="keep sw2robot's actuators (not passive)")
    ap.add_argument("--no-upright", action="store_true", help="don't rotate / place the model")
    ap.add_argument("--up", choices=["y", "hinge", "plate"], default="y",
                    help="which direction becomes vertical: SolidWorks Y (default), the hinge axis, or the base plate normal")
    ap.add_argument("--keep-contacts", action="store_true", help="don't exclude pairs touching in the CAD pose")
    ap.add_argument("--view", action="store_true", help="open the viewer after writing")
    ap.add_argument("--view-only", action="store_true", help="only view the given (already set up) file")
    ap.add_argument("--check", action="store_true", help="measure the hinge and handle joints of a *_sim.xml")
    ap.add_argument("--sweep", action="store_true", help="animate the door then the handle through their ranges")
    args = ap.parse_args()

    src = os.path.abspath(args.mjcf)
    workdir = os.path.dirname(src)
    cfg_path = os.path.join(workdir, "door_config.json")
    if args.view_only:
        view(src, cfg_path)
        return
    if args.check:
        check(src, cfg_path)
        return
    if args.sweep:
        sweep_view(src, cfg_path)
        return
    out = os.path.abspath(args.out) if args.out else os.path.splitext(src)[0] + "_sim.xml"

    tree = ET.parse(src)
    root = tree.getroot()
    comp = root.find("compiler")
    ang = (lambda r: r) if comp is not None and comp.get("angle") == "radian" else math.degrees
    m = compile_tree(tree, workdir)

    # ---- joints
    def joint_moving(key):
        b = body_by_key(m, key)
        js = [j for j in range(m.njnt) if m.jnt_bodyid[j] == b and int(m.jnt_type[j]) == HINGE]
        if not js:
            sys.exit(f"the '{key}' part has no hinge joint - check the joints in the sw2robot yaml")
        return m.joint(js[0]).name
    hinge = args.hinge or joint_moving("door")
    handle = args.handle or joint_moving("handle")
    jh, jd = m.joint(hinge).id, m.joint(handle).id
    for n, j in ((hinge, jh), (handle, jd)):
        lo, hi = m.jnt_range[j]
        lim = "limited" if m.jnt_limited[j] else "NO LIMITS"
        print(f"[joint] {n:<28} range [{math.degrees(lo):7.1f}, {math.degrees(hi):7.1f}] deg  ({lim})")
        if not m.jnt_limited[j] or hi - lo > 2 * math.pi - 0.01:
            print(f"        ! set real limits for {n} (LimitAngle) in the joints yaml and rebuild")

    # ---- masses
    for spec in args.mass:
        key, _, kg = spec.partition("=")
        kg = float(kg)
        b = body_by_key(m, key)
        el = find(root, "body", m.body(b).name)
        old = float(m.body_mass[b])
        f = kg / old if old > 0 else 1.0
        inert = el.find("inertial")
        if inert is None:
            inert = ET.SubElement(el, "inertial", pos=fmt(*m.body_ipos[b]), quat=fmt(*m.body_iquat[b]),
                                  diaginertia=fmt(*m.body_inertia[b]))
        inert.set("mass", fmt(kg))
        for attr in ("diaginertia", "fullinertia"):
            if inert.get(attr):
                inert.set(attr, fmt(*(float(x) * f for x in inert.get(attr).split())))
        print(f"[mass] {m.body(b).name}: {old:.3f} kg -> {kg:.3f} kg (inertia x{f:.3f})")

    # ---- passive joints: hinge damping/friction, handle spring
    eh, ed = find(root, "joint", hinge), find(root, "joint", handle)
    eh.set("damping", fmt(args.door_damping))
    eh.set("frictionloss", fmt(args.door_friction))
    eh.set("armature", "0.01")
    ed.set("stiffness", fmt(args.handle_spring))
    ed.set("springref", fmt(ang(float(m.qpos0[m.jnt_qposadr[jd]]))))
    ed.set("damping", fmt(args.handle_damping))
    ed.set("armature", "0.001")

    if not args.keep_actuators:
        for a in root.findall("actuator"):
            root.remove(a)
    for kf in root.findall("keyframe"):
        for key in kf:
            key.attrib.pop("ctrl", None)
            key.attrib.pop("act", None)

    # ---- latch (equality: hinge held at its CAD / closed position; switched at runtime)
    for eq in root.findall("equality"):
        for c in list(eq):
            if c.get("name") == "door_latch":
                eq.remove(c)
    use_eq_latch = not args.no_latch and (not args.handle_hits_frame or args.keep_latch)
    if use_eq_latch:
        eq = root.find("equality")
        if eq is None:
            eq = ET.SubElement(root, "equality")
        ET.SubElement(eq, "joint", name="door_latch", joint1=hinge, polycoef="0 0 0 0 0",
                      solref="0.01 1", active="true")

    # ---- sensors
    sens = root.find("sensor")
    if sens is None:
        sens = ET.SubElement(root, "sensor")
    names = {s.get("name") for s in sens}
    for nm, j in (("door_angle", hinge), ("handle_angle", handle)):
        if nm not in names:
            ET.SubElement(sens, "jointpos", name=nm, joint=j)

    opt = root.find("option")
    if opt is None:
        opt = ET.Element("option")
        root.insert(1 if comp is not None else 0, opt)
    opt.set("integrator", opt.get("integrator", "implicitfast"))

    # ---- upright: hinge axis -> +Z, frame above the base, base on z = 0
    m = compile_tree(tree, workdir)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    b = m.jnt_bodyid[jh]
    while m.body_parentid[b] != 0:
        b = m.body_parentid[b]
    rid = b
    subtree = {i for i in range(m.nbody) if i == rid or _ancestor(m, i, rid)}
    if not args.no_upright:
        # "up": SolidWorks Y (default - the export keeps the CAD frame), or the hinge axis,
        # or the base plate normal (largest-inertia axis; unreliable when the frame is merged in)
        if args.up == "y":
            axis, how = np.array([0.0, 1.0, 0.0]), "SolidWorks Y axis"
        elif args.up == "hinge":
            axis, how = np.array(d.xaxis[jh]), "door hinge axis"
        else:
            I = np.array(m.body_inertia[rid])
            axis = np.array(d.ximat[rid]).reshape(3, 3)[:, int(np.argmax(I))]
            how = "base plate normal"
        door_b = m.jnt_bodyid[jh]
        up_ref = np.array(d.subtree_com[door_b]) - np.array(d.xipos[rid])     # door centre vs base
        best = None
        for sgn in (1.0, -1.0):
            q = quat_from_to(sgn * axis, np.array([0, 0, 1.0]))
            score = rot(q, up_ref)[2]
            if best is None or score > best[0]:
                best = (score, q)
        q_add = best[1]
        el = find(root, "body", m.body(rid).name)
        for a in ("quat", "euler", "axisangle", "xyaxes", "zaxis"):
            el.attrib.pop(a, None)
        el.set("quat", fmt(*quat_mul(q_add, m.body_quat[rid])))
        el.set("pos", fmt(*m.body_pos[rid]))
        m = compile_tree(tree, workdir)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        z = lowest_z(m, d, subtree)
        el.set("pos", fmt(*(np.array(m.body_pos[rid]) + [0, 0, -z])))
        m = compile_tree(tree, workdir)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        hz = abs(float(d.xaxis[m.joint(hinge).id][2]))
        tilt = math.degrees(math.acos(min(1.0, hz)))
        print(f"[upright] stood up by the {how}")
        if tilt > 5:
            print(f"[WARNING] the door hinge axis is {tilt:.0f} deg from vertical - the hinge axis came out of the "
                  f"export wrong.\n          Fix: python drop_angle_mates.py $pkg ; python fix_axes.py $pkg --reset --write ; "
                  f"rebuild ; rerun door_setup.py")
        print(f"[upright] hinge axis now {np.round(d.xaxis[m.joint(hinge).id], 3).tolist()}, "
              f"base on z = {lowest_z(m, d, subtree) * 1000:.1f} mm, top of the model at "
              f"{z_extent(m, d, subtree)[1] * 1000:.0f} mm, handle pivot at "
              f"{d.xanchor[m.joint(handle).id][2] * 1000:.0f} mm")

    # ---- handle rests at its CAD pose: preload the return spring against the handle's own weight
    jd_ = m.joint(handle).id
    k = float(m.jnt_stiffness[jd_])
    if k > 0:
        d.qpos[:] = m.qpos0
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        g_torque = float(d.qfrc_bias[m.jnt_dofadr[jd_]])          # gravity torque on the handle at rest
        q0h = float(m.qpos0[m.jnt_qposadr[jd_]])
        sag = -g_torque / k
        find(root, "joint", handle).set("springref", fmt(ang(q0h - sag)))
        print(f"[handle] its weight would make it sag {math.degrees(sag):.1f} deg - spring preloaded so it rests level")
        m = compile_tree(tree, workdir)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)

    # ---- physical latch: frame as boxes with a pocket for the handle
    if args.handle_hits_frame:
        print(frame_boxes(root, m, d, hinge, handle, pocket=args.pocket))
        m = compile_tree(tree, workdir)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)

    # ---- exclude pairs touching in the CAD pose (door in frame, frame on base), and always the
    #      hinge pair frame <-> door: the hinge limits stop the door, and MuJoCo does not filter
    #      parent/child contacts when the parent (frame) is fixed to the world
    excluded = []
    cont0 = root.find("contact")
    if cont0 is not None:                                   # start from a clean list on every run
        for e in [e for e in cont0 if e.tag == "exclude"]:
            cont0.remove(e)
    if not args.keep_contacts:
        pairs = set()
        hb = m.jnt_bodyid[m.joint(hinge).id]
        pairs.add(tuple(sorted((m.body(m.body_parentid[hb]).name, m.body(hb).name))))
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if b1 and b2 and b1 != b2 and m.body(b1).name and m.body(b2).name:
                pairs.add(tuple(sorted((m.body(b1).name, m.body(b2).name))))
        if args.handle_hits_frame:
            hb_ = m.jnt_bodyid[m.joint(handle).id]
            fb_ = m.body_parentid[m.jnt_bodyid[m.joint(hinge).id]]
            keep = tuple(sorted((m.body(fb_).name, m.body(hb_).name)))
            pen = [float(d.contact[i].dist) for i in range(d.ncon)
                   if {int(m.geom_bodyid[d.contact[i].geom1]), int(m.geom_bodyid[d.contact[i].geom2])} == {hb_, fb_}]
            if keep in pairs:
                pairs.discard(keep)
            if pen and min(pen) < -0.001:
                where = sorted({m.geom(g).name or f"geom {g}" for i in range(d.ncon)
                                for g in (d.contact[i].geom1, d.contact[i].geom2)
                                if m.geom_bodyid[g] == fb_ and d.contact[i].dist < -0.001})
                print(f"[WARNING] the handle overlaps the frame by {-min(pen) * 1000:.1f} mm with the door closed "
                      f"(in: {', '.join(where)}) - it will be shoved out at the start. Press 3 in the viewer to see "
                      f"the red frame boxes.")
            else:
                print("[handle-frame] handle <-> frame collision ON, no overlap with the door closed")
        if pairs:
            cont = root.find("contact")
            if cont is None:
                cont = ET.SubElement(root, "contact")
            for n1, n2 in sorted(pairs):
                ET.SubElement(cont, "exclude", body1=n1, body2=n2)
                excluded.append(f"{n1} <-> {n2}")

    tree.write(out)
    m = mujoco.MjModel.from_xml_path(out)
    cfg = dict(model=os.path.basename(out), hinge=hinge, handle=handle,
               latch="door_latch" if use_eq_latch else None, unlock_deg=args.unlock_deg,
               physical_latch=bool(args.handle_hits_frame))
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print("\nparts:")
    for i in range(1, m.nbody):
        print(f"   {m.body(i).name:<16} {m.body_mass[i]:8.3f} kg")
    if excluded:
        print("excluded contact pairs touching in the CAD pose:", ", ".join(excluded))
    if args.handle_hits_frame:
        print("\nlatch: PHYSICAL - the handle's end stops the door against the frame until it is turned")
    print(f"{'constraint latch: door locked until the handle turns ' + str(args.unlock_deg) + ' deg' if use_eq_latch else 'no constraint latch'}")
    print(f"wrote {out}\nwrote {cfg_path}")

    if args.view:
        view(out, cfg_path)


def _ancestor(m, b, a):
    while b != 0:
        b = m.body_parentid[b]
        if b == a:
            return True
    return False


if __name__ == "__main__":
    main()
