#!/usr/bin/env python3
"""
make_standins.py - builds STAND-IN models of the hand and the door for testing the RL code without the CAD files.

They copy what matters about the real exports:
  hand : base_link (thick forearm) -> wrist_1 -> Palm_1 -> 5 fingers (Index/Middle/Ring/Pinky _Proximal/_Middle/
         _Distal, Thumb_Base/_Proximal/_Distal), exporter-style collision flags (contype 1, conaffinity 0)
  door : SolidWorks Y-up, base_link (floor plate + frame) -> door_1 (hinge) -> handle_1 (one MESH: long lever in
         front + spindle through the door + rod behind - its convex hull is a solid wedge, like the real one)
and are then built through the real pipeline: add_actuators.py (hand) and door_setup.py (door).

  python make_standins.py                 # -> standins/hand/Hand_actuated.xml, standins/door/Door_sim.xml
  python env_tests.py --quick --hand standins/hand/Hand_actuated.xml --door standins/door/Door_sim.xml
"""
import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def f(*v):
    return " ".join(f"{x:.5g}" for x in v)


# --------------------------------------------------------------------------- hand
def finger(name, x, lens, rad=0.0085):
    """name_Proximal_1 -> _Middle_1 -> _Distal_1, along +Z, flexing toward -Y (the palm side)."""
    parts = ("Proximal", "Middle", "Distal")
    xml, close = "", ""
    for k, (part, L) in enumerate(zip(parts, lens)):
        pos = f(x, 0, 0.097) if k == 0 else f(0, 0, lens[k - 1])
        rng = "-0.2 1.6" if k == 0 else "0 1.6"
        xml += (f'<body name="{name}_{part}_1" pos="{pos}">'
                f'<joint name="{name}_{part}_joint" type="hinge" axis="1 0 0" range="{rng}"/>'
                f'<geom name="{name}_{part}_geom" type="capsule" fromto="0 0 0 0 0 {L}" size="{rad}" '
                f'contype="1" conaffinity="0" density="1100" rgba="0.85 0.75 0.65 1"/>')
        close += "</body>"
    return xml + close


def hand_xml():
    b = np.array([-0.75, -0.15, 0.62])
    b /= np.linalg.norm(b)
    n = np.array([0.0, -1.0, 0.0])                       # palm side
    ax = np.cross(b, n)
    ax /= np.linalg.norm(ax)
    seg = lambda L: f(0, 0, 0, *(b * L))
    thumb = (f'<body name="Thumb_Base_1" pos="-0.034 -0.012 0.03">'
             f'<joint name="Thumb_Base_joint" type="hinge" axis="0 0 1" range="-0.2 1.3"/>'
             f'<geom name="Thumb_Base_geom" type="capsule" fromto="{seg(0.035)}" size="0.0105" contype="1" '
             f'conaffinity="0" density="1100" rgba="0.85 0.75 0.65 1"/>'
             f'<body name="Thumb_Proximal_1" pos="{f(*(b * 0.035))}">'
             f'<joint name="Thumb_Proximal_joint" type="hinge" axis="{f(*ax)}" range="0 1.4"/>'
             f'<geom name="Thumb_Proximal_geom" type="capsule" fromto="{seg(0.03)}" size="0.0095" contype="1" '
             f'conaffinity="0" density="1100" rgba="0.85 0.75 0.65 1"/>'
             f'<body name="Thumb_Distal_1" pos="{f(*(b * 0.03))}">'
             f'<joint name="Thumb_Distal_joint" type="hinge" axis="{f(*ax)}" range="0 1.4"/>'
             f'<geom name="Thumb_Distal_geom" type="capsule" fromto="{seg(0.025)}" size="0.009" contype="1" '
             f'conaffinity="0" density="1100" rgba="0.85 0.75 0.65 1"/>'
             f'</body></body></body>')
    fingers = (finger("Index", -0.03, (0.044, 0.026, 0.021)) + finger("Middle", -0.01, (0.048, 0.029, 0.022)) +
               finger("Ring", 0.01, (0.045, 0.027, 0.021)) + finger("Pinky", 0.03, (0.036, 0.021, 0.018)))
    return f'''<mujoco model="Hand_forearm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light pos="0 -1 2" dir="0 0.5 -1"/>
    <body name="base_link" pos="0 0 0">
      <geom name="forearm_geom" type="capsule" fromto="0 0 0.02 0 0 0.21" size="0.03" contype="1" conaffinity="0"
            density="900" rgba="0.6 0.6 0.65 1"/>
      <body name="wrist_1" pos="0 0 0.25">
        <joint name="wrist_joint" type="hinge" axis="0 1 0" range="-0.5 0.5"/>
        <geom name="wrist_geom" type="sphere" size="0.022" contype="1" conaffinity="0" density="900"
              rgba="0.6 0.6 0.65 1"/>
        <body name="Palm_1" pos="0 0 0.02">
          <joint name="Palm_joint" type="hinge" axis="1 0 0" range="-0.8 0.8"/>
          <geom name="Palm_geom" type="box" pos="0 0 0.05" size="0.043 0.010 0.046" contype="1" conaffinity="0"
                density="1000" rgba="0.85 0.75 0.65 1"/>
          {thumb}
          {fingers}
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
'''


# --------------------------------------------------------------------------- door (SolidWorks Y-up)
def boxes_mesh(boxes):
    """one mesh made of several boxes (lo, hi) - collides as ONE convex hull, like an exported part."""
    V, F = [], []
    for lo, hi in boxes:
        base = len(V)
        for i in range(8):
            V.append([hi[0] if i & 1 else lo[0], hi[1] if i & 2 else lo[1], hi[2] if i & 4 else lo[2]])
        for a, b_, c, d in ((0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)):
            F += [[base + a, base + b_, base + c], [base + a, base + c, base + d]]
    return " ".join(f"{x:.4g}" for v in V for x in v), " ".join(str(i) for t in F for i in t)


def door_xml():
    # handle (local frame: pivot on the door's front face; +Z = out of the door, lever toward the hinge = -X)
    lever = [((-0.17, -0.010, 0.056), (0.0, 0.010, 0.074)),      # long lever in front (20 x 18 mm)
             ((-0.016, -0.016, 0.0), (0.016, 0.016, 0.074)),     # neck / rose
             ((-0.005, -0.005, -0.055), (0.005, 0.005, 0.0)),    # spindle through the door (40 mm thick)
             ((-0.05, -0.006, -0.068), (0.0, 0.006, -0.055))]    # short stub behind
    v, fc = boxes_mesh(lever)
    return f'''<mujoco model="Full_door_v4">
  <compiler angle="radian" autolimits="true"/>
  <asset><mesh name="handle_mesh" vertex="{v}" face="{fc}"/></asset>
  <worldbody>
    <body name="base_link" pos="0 0 0">
      <geom name="plate" type="box" pos="0.4 -0.01 0" size="0.6 0.01 0.3" contype="1" conaffinity="0" rgba="0.5 0.5 0.5 1"/>
      <geom name="post_hinge" type="box" pos="-0.03 1.0 0" size="0.03 1.0 0.04" contype="1" conaffinity="0" rgba="0.55 0.4 0.3 1"/>
      <geom name="post_latch" type="box" pos="0.832 1.0 0" size="0.03 1.0 0.04" contype="1" conaffinity="0" rgba="0.55 0.4 0.3 1"/>
      <geom name="top" type="box" pos="0.4 2.03 0" size="0.46 0.03 0.04" contype="1" conaffinity="0" rgba="0.55 0.4 0.3 1"/>
      <body name="door_1" pos="0 0 0">
        <joint name="base_link__door_1" type="hinge" axis="0 1 0" range="0 1.6"/>
        <geom name="door_slab" type="box" pos="0.4 1.0 0.0" size="0.4 0.99 0.02" density="600" contype="1"
              conaffinity="0" rgba="0.7 0.55 0.4 1"/>
        <body name="handle_1" pos="0.72 1.0 0.02">
          <joint name="door_1__handle_1" type="hinge" axis="0 0 1" range="0 1.5708"/>
          <geom name="handle_geom" type="mesh" mesh="handle_mesh" density="7800" contype="1" conaffinity="0"
                rgba="0.8 0.8 0.85 1"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
'''


def run(cmd, cwd):
    r = subprocess.run([sys.executable] + cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit(f"failed: {' '.join(cmd)}")
    return r.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "standins"))
    a = ap.parse_args()
    hd, dd = os.path.join(a.out, "hand"), os.path.join(a.out, "door")
    os.makedirs(hd, exist_ok=True)
    os.makedirs(dd, exist_ok=True)
    open(os.path.join(hd, "Hand.xml"), "w").write(hand_xml())
    open(os.path.join(dd, "Door.xml"), "w").write(door_xml())
    out = run([os.path.join(HERE, "add_actuators.py"), "Hand.xml", "--fix-wrist",
               "--add-axis", "Thumb_Proximal_1@-45,45"], hd)
    print("[hand]", " ".join(l for l in out.splitlines() if "actuator" in l.lower())[:300])
    out = run([os.path.join(HERE, "door_setup.py"), "Door.xml", "--mass", "door=38.4", "--mass", "handle=0.5",
               ], dd)
    print("[door]", " | ".join(l.strip() for l in out.splitlines() if l.startswith(("[joint]", "[latch]", "[frame")))[:400])
    print(f"hand: {os.path.join(hd, 'Hand_actuated.xml')}\ndoor: {os.path.join(dd, 'Door_sim.xml')}")


if __name__ == "__main__":
    main()
