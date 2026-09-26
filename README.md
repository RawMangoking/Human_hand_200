# Human Hand + Door — SolidWorks, sw2robot, and MuJoCo Documentation

> **Project status:** CAD systems created in SolidWorks, hand and door exported through sw2robot, and a combined Hand + Door MuJoCo scene prepared.
>
> **Source of truth:** SolidWorks CAD defines the intended geometry, assembly relationships, reference geometry, and joint limits. MuJoCo is used for simulation physics, contacts, actuators, and control.

---

## 1. Project Overview

This project contains two main CAD systems:

1. A modular human hand with wrist/forearm support.
2. A physical-door-based assembly with a frame, door, base, and handle.

The intended final simulation is a single MuJoCo world in which the hand physically interacts with the active side of the door handle to operate the door.

High-level interaction:

```text
Hand
  ↓
Active Handle
  ↓
Door
  ↓
Hinge
  ↓
Frame
```

---

## 2. Repository Structure

Current CAD organization:

```text
Human_Hand_200mm/
├── 00_Master/
│   ├── Hand_Master_200mm.SLDPRT
│   └── Hand_Parameters.txt
├── 01_Palm/
│   └── Palm.SLDPRT
├── 02_Index/
│   ├── Index_Proximal.SLDPRT
│   ├── Index_Middle.SLDPRT
│   ├── Index_Distal.SLDPRT
│   └── index_assem.SLDASM
├── 03_Middle/
│   ├── Middle_Proximal.SLDPRT
│   ├── Middle_Middle.SLDPRT
│   ├── Middle_Distal.SLDPRT
│   └── Middle_Finger.SLDASM
├── 04_Ring/
│   ├── Ring_Proximal.SLDPRT
│   ├── Ring_Middle.SLDPRT
│   ├── Ring_Distal.SLDPRT
│   └── Ring_finger.SLDASM
├── 05_Pinky/
│   ├── Pinky_Proximal.SLDPRT
│   ├── Pinky_Middle.SLDPRT
│   ├── Pinky_Distal.SLDPRT
│   └── Pinky_finger.SLDASM
├── 06_Thumb/
│   ├── Thumb_Base.SLDPRT
│   ├── Thumb_Proximal.SLDPRT
│   ├── Thumb_Distal_v2.SLDPRT
│   └── Thumb_finger.SLDASM
├── 07_Assembly/
│   └── Human_Hand.SLDASM
├── Door/
│   └── Full_door_v3.SLDASM
├── Macro/
├── Mujoko/
│   ├── Hand_forearm_mjcf _v3/
│   └── Hand_Door/
└── README.md
```

The project also contains earlier hand assembly versions such as `human hand v1.SLDASM`, `human hand v2.SLDASM`, and `human hand v3.SLDASM`.

---

## 3. Coordinate-System Convention

This convention is important and must be kept consistent throughout the project.

### SolidWorks

```text
Vertical = Y
```

### sw2robot / MuJoCo

```text
Vertical = Z
```

Therefore:

```text
SolidWorks Y  →  MuJoCo Z
```

For this project, whenever **vertical** is mentioned in MuJoCo/sw2robot context, it means **MuJoCo Z**, even though the corresponding SolidWorks axis is Y.

MuJoCo gravity is configured along negative Z:

```xml
<option gravity="0 0 -9.81" />
```

---

# 4. Human Hand CAD

## 4.1 Overall Hand Size

The hand was designed around a target total length of:

```text
Wrist/palm reference → middle fingertip = 200 mm
```

Key palm dimensions:

```text
Palm width  = 89 mm
Palm length = 104 mm
Palm extrusion = 18 mm (Mid Plane)
```

The middle finger was dimensioned so that the total hand length is exactly 200 mm:

```text
104 + 45 + 27 + 24 = 200 mm
```

### Middle finger

```text
Proximal = 45 mm   width 17 → 15 mm
Middle   = 27 mm   width 15 → 13 mm
Distal   = 24 mm   width 13 → 12 mm
```

### Index finger

```text
Proximal = 39 mm   width 16 → 14 mm
Middle   = 22 mm   width 14 → 13 mm
Distal   = 20 mm   width 13 → 12 mm
```

### Ring finger

```text
Proximal = 41 mm   width 16 → 14 mm
Middle   = 25 mm   width 14 → 13 mm
Distal   = 20 mm   width 13 → 12 mm
```

### Pinky finger

```text
Proximal = 32 mm   width 14 → 12 mm
Middle   = 18 mm   width 12 → 11 mm
Distal   = 18 mm   width 11 → 10 mm
```

### Thumb

Approximate geometry dimensions:

```text
Base/metacarpal = 45 mm
Proximal        ≈ 31 mm
Distal          ≈ 27 mm
Widths          ≈ 18 → 15 → 13 → 11 mm
```

Finger gaps were deliberately not finalized as an independent dimensional parameter because the total 200 mm hand-length requirement was the controlling constraint.

---

## 4.2 Palm Reference Geometry

MCP reference points are arranged on the straight upper palm section.

Distances from the left tangent used for the four MCP centers:

```text
Index = 8.5 mm
Middle = 28.0 mm
Ring = 47.5 mm
Pinky = 65.5 mm
```

Thumb CMC reference:

```text
Thumb CMC point = 70 mm below the top/MCP reference
```

Thumb CMC axes:

```text
Thumb_CMC_Axis_1
Thumb_CMC_Axis_2
```

These can support a two-DOF simplified CMC implementation if required.

Wrist reference:

```text
Wrist point = center of the short lower wrist edge
Wrist_Axis  = transverse wrist axis
```

---

# 5. Human Hand SolidWorks Mate Hierarchy

The complete recursive SolidWorks mate-tree extraction identified **43 hand mate features** across the main hand assembly and the five nested finger/thumb subassemblies.

| Assembly | Mate count |
|---|---:|
| Main `human hand v3` assembly | 15 |
| Index | 6 |
| Middle | 6 |
| Ring | 6 |
| Pinky | 6 |
| Thumb | 4 |
| **Total** | **43** |

The mate types in the macro output are `MateCoincident` for Coincident mates and `MateLimitPlanarAngleDim` for LimitAngle mates.

## 5.1 Main hand assembly

Assembly:

```text
human hand v3.SLDASM
```

Mates:

```text
Coincident2
Coincident3
Coincident4
Coincident5
Coincident7
LimitAngle1
Coincident9
LimitAngle2
Coincident11
LimitAngle3
Coincident13
LimitAngle4
Coincident14
LimitAngle10
Coincident17
```

These connect the Palm to the four finger subassemblies and thumb subassembly.

The direct main-assembly components are:

```text
Palm-1
index_assem-2
Middle_Finger-1
Ring_finger-1
Pinky_finger-1
Thumb_finger-1
```

## 5.2 Index assembly

Assembly:

```text
02_Index/index_assem.SLDASM
```

Components:

```text
Index_Proximal-1
Index_Middle-1
Index_Distal-1
```

Mates:

```text
Coincident1
Coincident2
LimitAngle2
Coincident4
LimitAngle5
Coincident6
```

The documented kinematic intent is:

```text
Palm
  ↓
Index Proximal
  ↓ MCP
Index Middle
  ↓ PIP
Index Distal
  ↓ DIP positioning
```

Explicit angular-limit mates present in the CAD assembly:

```text
LimitAngle2
LimitAngle5
```

A dedicated DIP angular-limit mate was not separately documented.

## 5.3 Middle assembly

Assembly:

```text
03_Middle/Middle_Finger.SLDASM
```

Components:

```text
Middle_Proximal-5
Middle_Middle-4
Middle_Distal-4
```

Mates:

```text
Coincident11
Coincident13
LimitAngle2
Coincident15
LimitAngle4
Coincident18
```

## 5.4 Ring assembly

Assembly:

```text
04_Ring/Ring_finger.SLDASM
```

Components:

```text
Ring_Proximal-1
Ring_Middle-1
Ring_Distal-1
```

Mates:

```text
Coincident1
Coincident2
LimitAngle1
Coincident3
Coincident4
LimitAngle2
```

## 5.5 Pinky assembly

Assembly:

```text
05_Pinky/Pinky_finger.SLDASM
```

Components:

```text
Pinky_Proximal-1
Pinky_Middle-1
Pinky_Distal-1
```

Mates:

```text
Coincident1
Coincident2
LimitAngle1
Coincident3
Coincident4
LimitAngle2
```

## 5.6 Thumb assembly

Assembly:

```text
06_Thumb/Thumb_finger.SLDASM
```

Components:

```text
Thumb_Base-1
Thumb_Proximal-1
Thumb_Distal_v2-1
```

Mates:

```text
Coincident2
Coincident3
LimitAngle1
Coincident4
```

Thumb CMC references:

```text
Thumb_CMC_Axis_1
Thumb_CMC_Axis_2
```

---

# 6. Integrated Hand + Wrist + Forearm CAD

Integrated assembly:

```text
Hand_forearm.SLDASM
```

Top-level direct components:

```text
wrist-1
human hand v3-2
```

Top-level mates from the complete recursive macro:

```text
Coincident5
Coincident6
Coincident7
LimitAngle3
LimitAngle5
```

Total top-level integrated mates:

```text
5
```

The integrated structure is intended as:

```text
Forearm
  ↓
Wrist
  ↓
Palm
  ↓
Hand
```

For the initial MuJoCo implementation, the wrist/forearm is treated as mounting/support structure rather than adding a new wrist control DOF.

The detailed meanings of the integrated `LimitAngle3` and `LimitAngle5` were not finalized as independent wrist DOFs.

---

# 7. Human Hand MuJoCo Export

The current correct hand MJCF is based on:

```text
Hand_forearm
```

Current export characteristics:

```text
18 links
15 movable hinge joints
15 position actuators
25 meshes verified during export
24 meshes exported in the extraction record
```

The current MJCF keeps the SolidWorks-derived body transforms and joint axes generated by the latest sw2robot extraction.

The root body is currently represented by `base_link`; the hand/fingers are nested under that body. The current XML contains 15 hinge joints and 15 position actuators. fileciteturn13file0L48-L56 fileciteturn13file0L149-L167

---

# 8. Human Hand MuJoCo Control Design

The physical simulation keeps the five fingers separate, but the controller is intended to expose three high-level control groups.

## Group 1 — Index

```text
Index_MCP
Index_PIP
Index_DIP
```

## Group 2 — Middle + Ring + Pinky

Corresponding commands are shared:

```text
middle_MCP = ring_MCP = pinky_MCP
middle_PIP = ring_PIP = pinky_PIP
middle_DIP = ring_DIP = pinky_DIP
```

The three fingers remain separate physical bodies.

## Group 3 — Thumb

```text
Thumb_CMC
Thumb_MCP
Thumb_IP
```

The thumb CMC may be implemented with one or two rotational DOFs depending on the final controller design.

---

# 9. Door CAD

Current SolidWorks assembly:

```text
Door/Full_door_v3.SLDASM
```

Main components extracted by sw2robot:

```text
base-1
frame-1
door-1
handle-1
```

The SolidWorks model is based on the physical door.

## 9.1 Door dimensions used for mass sanity check

Door-only dimensions supplied for the simulation:

```text
Length    = 2.00 m
Width     = 0.80 m
Thickness = 0.04 m
```

Volume:

```text
V = 2.00 × 0.80 × 0.04
  = 0.064 m³
```

Custom wood density selected in sw2robot:

```text
600 kg/m³
```

Door mass check:

```text
m = 0.064 × 600
  = 38.4 kg
```

The current door MJCF contains:

```xml
mass="38.4"
```

for `door_1`.

The earlier ~930 kg value belonged to `base_1`, not to the physical door slab. At 1240 kg/m³, 930 kg corresponds to approximately 0.75 m³ of CAD volume. That component must therefore be treated as the large fixed base/structural geometry rather than being interpreted as the door slab mass.

---

# 10. Door Materials

sw2robot did not provide a preset called `Wood`, so a custom material/density was used.

Current density configuration used for the door export:

```yaml
densities:
  handle_1: 600
  door_1: 600
  frame_1: 600
  base_1: 600
```

The 600 kg/m³ value is the simulation approximation selected for wood in this project.

The actual `handle_1` material should correspond to the physical handle if a more accurate mass model is required later.

The current exported door XML has the following dynamic masses:

```text
door_1   = 38.4 kg
handle_1 = 8.4278 kg
```

The fixed/root structure is substantially heavier because it includes the frame/base geometry.

---

# 11. Door Reference Geometry

The SolidWorks door/frame assembly contains dedicated reference geometry that was inspected while diagnosing the exported joint axes.

## Frame references

```text
frame hinge
Point1
Axis2
Point2
Axis3
Point3
Axis4
Plane1
Point4
Axis5
```

## Door-side hinge references

```text
hinge
Point1
Point2
Axis2
Axis3
Axis4
Point3
Axis5
Axis6
```

## Handle references

```text
Handle center
Axis1
Axis2
```

These references are preferable to arbitrary mesh coordinates when a future joint needs to be reconstructed manually in MuJoCo.

---

# 12. Door SolidWorks Mate Hierarchy

The top-level door mate-tree extraction found **10 mates**.

Assembly:

```text
Door/Full_door_v3.SLDASM
```

Mate group:

```text
Mates
```

Exact mate list:

```text
Coincident2
Coincident3
Coincident4
Coincident5
Coincident10
Coincident11
LimitAngle2
LimitAngle3
Coincident12
Coincident13
```

Mate types:

```text
Coincident*  → MateCoincident
LimitAngle*  → MateLimitPlanarAngleDim
```

---

# 13. Detailed Door Mate-Entity Macro Results

A second SolidWorks VBA macro was used to inspect each `Mate2`, its `MateEntity2` objects, `ReferenceComponent`, `ReferenceType`, and `EntityParams`.

`EntityParams` was interpreted in the report as:

```text
[0] X
[1] Y
[2] Z
[3] Vector X / I
[4] Vector Y / J
[5] Vector Z / K
[6] Radius1
[7] Radius2
```

Coordinates in the report were converted to millimetres for readability.

The detailed output supplied for the current door report begins at `Coincident3`; `Coincident2` is confirmed in the mate-tree list above, but its detailed entity block was not included in the pasted report. Therefore no values for `Coincident2` are invented here.

## Coincident3

```text
ENTITY 0
Component: base-1
Reference type: 3
Point (mm): (-863.798545, 2236.424706, 6123.121076)
Vector: (0, 1, 0)

ENTITY 1
Component: frame-1
Reference type: 1
Point (mm): (1100.350043, 2236.424706, 3723.899707)
Vector: (1, 0, 0)
```

## Coincident4

```text
ENTITY 0
Component: base-1
Reference type: 3
Point (mm): (-863.798545, 2236.424706, 6123.121076)
Vector: (0, 1, 0)

ENTITY 1
Component: frame-1
Reference type: 1
Point (mm): (1100.350043, 2236.424706, 3723.899707)
Vector: (0, 0, -1)
```

## Coincident5

```text
ENTITY 0
Component: base-1
Reference type: 1
Point (mm): (-939.649957, 2236.424706, 3703.899707)
Vector: (1, 0, 0)

ENTITY 1
Component: frame-1
Reference type: 1
Point (mm): (1560.350043, 2236.424706, 3703.899707)
Vector: (1, 0, 0)
```

## Coincident10

```text
ENTITY 0
Component: door-1
Reference type: 0
Point (mm): (1322.498557, 3242.175566, 3656.593820)
Vector: (1, 0, 0)

ENTITY 1
Component: handle-1
Reference type: 0
Point (mm): (1322.498557, 3242.175566, 3656.593820)
Vector: (1, 0, 0)
```

## Coincident11

```text
ENTITY 0
Component: door-1
Reference type: 1
Point (mm): (1322.498557, 3242.175566, 3656.593820)
Vector: (-0.072825756, 0, 0.997344679)

ENTITY 1
Component: handle-1
Reference type: 1
Point (mm): (1322.498557, 3242.175566, 3656.593820)
Vector: (-0.072825756, 0, 0.997344679)
```

## LimitAngle2

```text
ENTITY 0
Component: door-1
Reference type: 1
Point (mm): (1970.350043, 4248.670834, 3703.899707)
Vector: (0.997344679, 0, 0.072825756)

ENTITY 1
Component: frame-1
Reference type: 1
Point (mm): (1970.350043, 4248.670834, 3703.899707)
Vector: (1, 0, 0)
```

## LimitAngle3

```text
ENTITY 0
Component: door-1
Reference type: 1
Point (mm): (1172.474300, 4248.670834, 3645.639102)
Vector: (0.997344679, 0, 0.072825756)

ENTITY 1
Component: handle-1
Reference type: 1
Point (mm): (1322.498557, 3242.175566, 3656.593820)
Vector: (0, 1, 0)
```

### Handle-axis result

The detailed SolidWorks mate data therefore records the handle-side entity of `LimitAngle3` as:

```text
SolidWorks handle axis/vector = (0, 1, 0)
```

Under the project's coordinate convention this is the **SolidWorks vertical/Y direction**, which corresponds to **MuJoCo vertical/Z** after the project's coordinate-system conversion.

The handle rotation point extracted by the macro is:

```text
(1322.498557, 3242.175566, 3656.593820) mm
```

This data was used to diagnose the handle joint orientation exported by sw2robot.

## Coincident12

```text
ENTITY 0
Component: frame-1
Reference type: 0
Point (mm): (1970.350043, 4248.670834, 3703.899707)
Vector: (1, 0, 0)

ENTITY 1
Component: door-1
Reference type: 0
Point (mm): (1970.350043, 4248.670834, 3703.899707)
Vector: (1, 0, 0)
```

## Coincident13

```text
ENTITY 0
Component: frame-1
Reference type: 1
Point (mm): (1970.350043, 4261.158795, 3703.899707)
Vector: (0, 1, 0)

ENTITY 1
Component: door-1
Reference type: 1
Point (mm): (1970.350043, 4248.670834, 3703.899707)
Vector: (0, 1, 0)
```

---

# 14. Door Kinematic Structure

Intended physical structure:

```text
Frame [fixed]
   │
   └── Door [hinge rotation]
           │
           └── Handle [lever rotation]
```

The frame is fixed.

The door rotates around the physical hinge axis.

The handle is separate so its lever motion can be represented in MuJoCo.

---

# 15. Door Joint Configuration in sw2robot

The current `Full_door_v3.joints.yaml` contains:

```yaml
base: frame_1
joints:
  - parent: frame_1
    child:  base_1
    type:   fixed

  - parent: frame_1
    child:  door_1
    type:   revolute
    lower: -1.64369
    upper: 1.49791

  - parent: door_1
    child:  handle_1
    type:   revolute
    lower: -1.57080
    upper: 0.00000
```

Therefore the intended exporter-side joint limits are:

```text
Door hinge   = -1.64369 → +1.49791 rad
Handle       = -1.57080 → 0 rad
```

Approximate degrees:

```text
Door hinge   ≈ -94.18° → +85.85°
Handle       = -90° → 0°
```

The YAML is also configured with:

```yaml
root_xyz: [0, 0, 0]
root_rpy: [1.67887e-11, 6.32679e-06, 5.30718e-06]
```

The axis itself is generated from the CAD/mate information rather than being manually authored in this YAML.

---

# 16. sw2robot Door Extraction

The successful door extraction produced:

```text
4 components found
2 limit-mate joints found
4 component meshes exported
5 meshes verified including the full-assembly mesh
4 links in the URDF
2 movable joints
```

The extraction reused:

```text
Full_door_v3.joints.yaml
```

for the fixed/revolute topology and joint limits.

The sw2robot URDF graph is:

```text
base/root
├── base_1       fixed
└── door_1       revolute
      └── handle_1  revolute
```

---

# 17. Current Door MuJoCo XML

Current model name:

```text
Full_door_v3
```

Current door MJCF uses:

```xml
<compiler angle="radian" autolimits="true" meshdir="assets" />
<option gravity="0 0 -9.81" />
```

The current root body was rotated so the door is vertical in MuJoCo:

```xml
<body name="base_link"
      pos="0 0 0.09"
      euler="1.57079632679 0 0">
```

That is a +90° X rotation.

The current exported door joint definitions are:

```xml
<joint name="frame_1__door_1"
       pos="0 0 0"
       axis="0 -1 0"
       type="hinge"
       range="-1.64369 1.49791"
       damping="3.18471338" />
```

and:

```xml
<joint name="door_1__handle_1"
       pos="0 0 0"
       axis="0.875955702 0 0.482391551"
       type="hinge"
       range="-1.5708 0"
       damping="3.18471338" />
```

The latest door XML supplied for the project should be treated as the current validated starting point.

---

# 18. Door Collision Strategy

The CAD door can geometrically intersect/overlap the frame in the SolidWorks representation. That overlap should not automatically become a hard MuJoCo collision constraint that blocks the intended hinge motion.

Important intended collision behavior:

```text
Hand ↔ Active Handle     ENABLE
Hand ↔ Door              ENABLE where physically relevant
Hand ↔ Frame             ENABLE where physically relevant
Door ↔ Frame              selectively configured
Handle ↔ Frame            must not pass through frame
```

Only one side of the door is the active interaction side for the current task. The opposite side should not introduce unnecessary control/collision complexity in the first simulation.

---

# 19. Door Push/Pull Task

The active handle is intended to support both:

```text
Push door
Pull door
```

Interaction chain:

```text
Hand
  ↓
Active Handle
  ↓
Door
  ↓
Hinge
  ↓
Frame
```

---

# 20. Combined Hand + Door MuJoCo Scene

The intended final scene is one MuJoCo world containing both articulated systems.

```text
MUJOCO WORLD
│
├── Door
│   ├── Frame [fixed]
│   └── Door [hinge]
│       └── Active Handle [hinge]
│
└── Hand / Forearm
    └── Palm
        ├── Index
        ├── Middle
        ├── Ring
        ├── Pinky
        └── Thumb
```

High-level hand controller groups:

```text
Control 1 → Index
Control 2 → Middle + Ring + Pinky
Control 3 → Thumb
```

Door controls:

```text
Door hinge
Handle lever
```

The combined scene is intended to place the hand near the active handle so contact can later be used for a physical door-manipulation task.

---

# 21. Current Combined MJCF

A combined MJCF was prepared separately from the original hand and door exports so that the individual exports remain available as references.

Recommended organization:

```text
Mujoko/
├── Hand_forearm_mjcf _v3/
│   └── Hand_forearm_mjcf/
│       ├── mjcf/
│       │   └── Hand_forearm.xml
│       └── assets/
│
└── Hand_Door/
    ├── Hand_Door.xml
    └── assets/
        ├── hand/
        └── door/
```

The combined model should use unique root-body names, for example:

```text
door_base
hand_base
```

instead of having two bodies both called `base_link`.

The combined simulation contains:

```text
15 hand joints
2 door joints
17 total joints

15 hand actuators
2 door actuators
17 total actuators
```

---

# 22. MuJoCo Mesh and Asset Rules

Do not mix the hand and door STL folders accidentally. The combined XML should reference the matching mesh files in its asset directories.

Recommended structure:

```text
assets/
├── hand/
│   ├── fore_arm_1_0.stl
│   ├── wrist_1_1.stl
│   ├── human_hand_v3__Palm_1_2.stl
│   ├── ...
│   └── Ring_finger__Ring_Distal_1_35.stl
│
└── door/
    ├── frame_1_0.stl
    ├── base_1_1.stl
    ├── frame_1_2.stl
    ├── base_1_3.stl
    ├── door_1_4.stl
    ├── door_1_5.stl
    ├── handle_1_6.stl
    └── handle_1_7.stl
```

The current hand XML itself contains 36 mesh assets and the current door XML contains 8 mesh assets.

---

# 23. Physics and Actuator Settings

Current hand export uses position actuators with:

```text
kp = 50
forcerange = -10 10
ctrlrange follows joint range
```

The door export uses the same basic position-actuator structure:

```xml
<position ... kp="50" ... forcerange="-10 10" />
```

Current door joint damping is:

```text
3.18471338
```

These are simulation starting values and can be tuned later for contact stability and task control.

---

# 24. SolidWorks Macro Work Completed

## Macro 1 — Recursive hand/forearm mate tree

Purpose:

```text
Traverse the MateGroup recursively
Identify nested subassemblies
List every mate and mate type
List the component hierarchy
```

Result:

```text
5 top-level Hand_forearm mates
43 hand/finger/thumb mates
```

The complete hand/forearm report contained:

```text
Hand_forearm.SLDASM
├── 5 top-level mates
└── human hand v3.SLDASM
    ├── 15 main-hand mates
    ├── 6 Index mates
    ├── 6 Pinky mates
    ├── 6 Ring mates
    ├── 4 Thumb mates
    └── 6 Middle mates
```

## Macro 2 — Detailed door mate/entity extraction

Purpose:

```text
Find the MateGroup
Read each Mate2
Read GetMateEntityCount
Read each MateEntity
Read ReferenceComponent
Read ReferenceType
Read EntityParams
```

This macro established the detailed geometry vectors used to diagnose the door hinge and handle axes.

---

# 25. CAD Mates vs MuJoCo Joints

SolidWorks mates should not be copied one-to-one into MuJoCo.

SolidWorks provides:

```text
Positioning
Alignment
Orientation
Assembly relationships
Angular limits
```

MuJoCo requires:

```text
Bodies
Joints
Joint frames
Joint axes
Joint limits
Mass/inertia
Collision geometry
Contact parameters
Actuators
```

Therefore:

```text
SolidWorks = source of mechanical intent
MuJoCo     = simulation implementation
```

---

# 26. Current Project Status

## Human Hand

- [x] Master hand planning completed
- [x] Palm modeled
- [x] Palm reference geometry established
- [x] MCP reference points created
- [x] Wrist reference created
- [x] Thumb CMC point created
- [x] Thumb CMC two-axis reference structure created
- [x] Index geometry modeled
- [x] Middle geometry modeled
- [x] Ring geometry modeled
- [x] Pinky geometry modeled
- [x] Thumb geometry modeled
- [x] Finger subassemblies created
- [x] Main hand assembly created
- [x] Complete 43-mate hand hierarchy extracted
- [x] Wrist/forearm assembly created
- [x] 5 integrated hand/forearm top-level mates extracted
- [x] Hand exported through sw2robot
- [x] Current correct hand MJCF established
- [x] 15 hand joints present
- [x] 15 hand actuators present
- [x] Three high-level control groups defined

## Door

- [x] Physical-door-based CAD created
- [x] Frame modeled
- [x] Base modeled
- [x] Door modeled
- [x] Handle modeled
- [x] Hinge references created
- [x] Handle references created
- [x] Top-level door mate tree extracted
- [x] 10 top-level door mates identified
- [x] Detailed mate/entity macro created
- [x] Detailed `LimitAngle2` data extracted
- [x] Detailed `LimitAngle3` handle data extracted
- [x] Handle SolidWorks mate axis identified as `(0, 1, 0)`
- [x] Door-only dimensions recorded: 2.0 × 0.8 × 0.04 m
- [x] Custom wood density selected: 600 kg/m³
- [x] Door mass verified as 38.4 kg
- [x] Door exported through sw2robot
- [x] Door hinge joint exported
- [x] Handle joint exported
- [x] Door root rotated +90° about X so door is vertical in MuJoCo
- [x] Current door XML established

## Combined simulation

- [x] Combined Hand + Door MJCF prepared
- [x] Unique root-body naming planned
- [x] Hand and door actuators combined
- [ ] Final hand placement relative to handle
- [ ] Verify hand/handle contact in the combined scene
- [ ] Tune collision filters
- [ ] Tune contact/friction parameters
- [ ] Verify door hinge behavior under contact forces
- [ ] Implement high-level 3-group hand controller
- [ ] Implement door manipulation controller
- [ ] Train/validate RL interaction task

---

# 27. Git History / Project Milestones

Known commits already made in the repository:

```text
 e38df3c  Initial folder and file creation
 a483b05  designed all fingers and began assembly
 27eb532  thumb redesigned and assembled
 b5eacb0  forgot to save
 2ae01d0  macro + README
 89536b7  add gitignore
 c38dd04  updated readme for door
```

The repository uses:

```text
remote = https://github.com/RawMangoking/Human_hand_200.git
branch = main
```

The `.gitignore` was added so SolidWorks temporary files such as `~$...` do not remain tracked.

---

# 28. Recommended Working Method Going Forward

Use SolidWorks/sw2robot when changing:

```text
Geometry
Part sizes
Assembly structure
Reference geometry
Mechanical mates
Joint topology
```

Edit MuJoCo MJCF directly for simulation-specific changes such as:

```text
World placement
Body orientation
Joint damping
Joint limits when intentionally overridden
Actuators
Control gains
Collision filtering
Contact settings
Friction
Initial keyframes
High-level controller logic
```

Keep the clean SolidWorks-derived exports untouched as rollback references whenever possible.

---

# 29. Current Source Files / Important Artifacts

Hand:

```text
Hand_forearm.SLDASM
Hand_forearm.xml
```

Door:

```text
Door/Full_door_v3.SLDASM
Full_door_v3.joints.yaml
Full_door_v3.xml
```

Combined simulation:

```text
Hand_Door.xml
```

Mate documentation/macros:

```text
Complete Hand + Forearm Mate Tree
Detailed Door Mate/Entity Report
```

---

# 30. Final System Picture

```text
                         MUJOCO WORLD
                               │
            ┌──────────────────┴──────────────────┐
            │                                     │
          HAND                                   DOOR
            │                                     │
       Forearm/Wrist                         Frame [fixed]
            │                                     │
          Palm                              Door [hinge]
       ┌────┼────┬────┐                           │
    Index Middle Ring Pinky                      Handle
       │      │     │                              │
       └──────┴─────┘                              │
            │                                      │
          Thumb                                    │
            │                                      │
            └──────────── contact ─────────────────┘
```

The final objective is a physically simulated hand interacting with the door rather than a purely visual animation.

---

## 31. Important Notes for Future Changes

1. Keep the coordinate convention explicit:

```text
SolidWorks vertical = Y
MuJoCo vertical      = Z
```

2. Do not judge door mass from `base_1`; the physical door slab is `door_1`, whose current 2.0 × 0.8 × 0.04 m geometry at 600 kg/m³ gives 38.4 kg.

3. Do not rotate the whole MuJoCo scene to fix a single joint-axis problem. Change the affected body/joint frame only.

4. The SolidWorks handle mate data is the authoritative reference used when diagnosing the handle-axis problem.

5. The current `Hand_forearm.xml` is the correct hand XML reference for the project; do not replace it with older manually modified variants unless intentionally revisiting the orientation decision.

6. The combined `Hand_Door.xml` should remain a separate simulation-specific file so the clean hand and door exports can always be regenerated independently.
