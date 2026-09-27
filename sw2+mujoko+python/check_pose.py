#!/usr/bin/env python3
"""
check_pose.py
=============
Where does a misalignment between two parts come from?

Compares every parent -> child part pair in
  (A) the raw sw2robot export  (Hand_forearm.xml, joints at their CAD pose)
  (B) the actuated model       (Hand_forearm_actuated.xml, in the controller's START pose)
and prints how much the child is rotated relative to its parent in each, split into
  bend   : rotation about the joint's own hinge axis(es)   (expected: fingers un-bend to straight)
  other  : any rotation that is NOT about a hinge axis    (a TWIST - should be ~0 everywhere)

Usage:
  python check_pose.py <folder with Hand_forearm.xml, Hand_forearm_actuated.xml, hand_config.json>
  python check_pose.py <mjcf folder> --pair Thumb_Base Thumb_Proximal     # details for one pair
"""
import argparse
import json
import math
import os

import mujoco
import numpy as np


def rel_rot(d, parent, child):
    Rp = d.xmat[parent].reshape(3, 3)
    Rc = d.xmat[child].reshape(3, 3)
    return Rp.T @ Rc


def angle(R):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="the mjcf folder")
    ap.add_argument("--raw", default="Hand_forearm.xml")
    ap.add_argument("--actuated", default="Hand_forearm_actuated.xml")
    ap.add_argument("--pair", nargs=2, metavar=("PARENT", "CHILD"), help="parts of the two body names")
    args = ap.parse_args()

    f = os.path.abspath(args.folder)
    ma = mujoco.MjModel.from_xml_path(os.path.join(f, args.raw))
    mb = mujoco.MjModel.from_xml_path(os.path.join(f, args.actuated))
    cfg = json.load(open(os.path.join(f, "hand_config.json")))
    da, db = mujoco.MjData(ma), mujoco.MjData(mb)
    mujoco.mj_forward(ma, da)                                    # raw: CAD pose
    for n, j in cfg["joints"].items():                           # actuated: controller start pose
        jid = mujoco.mj_name2id(mb, mujoco.mjtObj.mjOBJ_JOINT, n)
        db.qpos[mb.jnt_qposadr[jid]] = j["open"] if j["role"] == "flex" else j["neutral"]
    mujoco.mj_forward(mb, db)

    print(f"\n{'parent -> child':<78}{'CAD (raw)':>10}{'start pose':>12}{'change':>9}{'TWIST':>8}")
    for b in range(1, mb.nbody):
        p = mb.body_parentid[b]
        if p == 0:
            continue
        nb, np_ = mb.body(b).name, mb.body(p).name
        if args.pair and not (args.pair[0].lower() in np_.lower() and args.pair[1].lower() in nb.lower()):
            continue
        ia = mujoco.mj_name2id(ma, mujoco.mjtObj.mjOBJ_BODY, nb)
        pa = mujoco.mj_name2id(ma, mujoco.mjtObj.mjOBJ_BODY, np_)
        if ia < 0 or pa < 0:
            continue
        Ra, Rb = rel_rot(da, pa, ia), rel_rot(db, p, b)
        D = Ra.T @ Rb                                            # change in the child's frame
        change = angle(D)
        # remove the part of the change that is a rotation about this body's hinge axes
        axes = [mb.jnt_axis[j] for j in range(mb.njnt) if mb.jnt_bodyid[j] == b and mb.jnt_type[j] == 3]
        w = np.array([D[2, 1] - D[1, 2], D[0, 2] - D[2, 0], D[1, 0] - D[0, 1]]) / 2   # ~ axis * sin(angle)
        for a in axes:
            a = np.asarray(a) / np.linalg.norm(a)
            w = w - a * (w @ a)
        twist = math.degrees(math.asin(min(1.0, np.linalg.norm(w))))
        flag = "   <-- not about a hinge" if twist > 1.0 else ""
        label = f"{np_[-36:]} -> {nb[-36:]}"
        print(f"{label:<78}{angle(Ra):>9.1f}°{angle(Rb):>11.1f}°{change:>8.1f}°{twist:>7.1f}°{flag}")
    print("\nCAD (raw)  : child rotation relative to parent as exported by sw2robot (the extraction snapshot)")
    print("start pose : the same in the actuated model at the controller's start pose")
    print("TWIST      : part of the change NOT about the joint's hinge axes (should be ~0)")


if __name__ == "__main__":
    main()
