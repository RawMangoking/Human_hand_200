# Human Hand + Wrist + Forearm + Door — CAD and MuJoCo Documentation

## 1. Project Overview

The project contains a modular SolidWorks model of a human hand, a wrist/forearm mounting structure, and a physical-door-based CAD model. The intended next stage is a MuJoCo scene containing the hand interacting with the door.

The CAD is being used as the source for geometry, reference axes, assembly relationships, and joint limits. MuJoCo will provide the final physics, contact, and control implementation.

---

# 2. Current CAD Structure

The hand remains a modular five-finger model:

```text
Palm
├── Index
├── Middle
├── Ring
├── Pinky
└── Thumb
```

The new integrated assembly adds a wrist and forearm:

```text
Forearm
  ↓
Wrist
  ↓
Palm
  ↓
Hand
```

Integrated assembly:

```text
Hand_forearm.SLDASM
├── wrist-1
└── human hand v3-2
```

The latest SolidWorks mate-tree extraction identifies the integrated assembly file as `Hand_forearm.SLDASM` and the two direct top-level components as `wrist-1` and `human hand v3-2`. fileciteturn1file0L5-L32

---

# 3. Wrist + Forearm Assembly

## Integrated Assembly

File:

```text
Hand_forearm.SLDASM
```

Top-level mates:

```text
Coincident5
Coincident6
Coincident7
LimitAngle3
LimitAngle5
```

Total top-level mates in the integrated hand/forearm assembly: **5**. The extracted report shows these five mates before the direct `wrist-1` and `human hand v3-2` components and before entering the nested hand assembly. fileciteturn1file0L9-L32

## MuJoCo intent

The forearm is intended to act as the fixed support/base for the first MuJoCo version:

```text
world
  ↓
forearm       fixed
  ↓
wrist
  ↓
palm
  ↓
hand
```

For the initial simulation, the wrist/forearm structure should be treated as the mounting structure rather than introducing a new wrist control DOF. The exact meaning and limits of `LimitAngle3` and `LimitAngle5` should be verified from detailed mate-entity data before deciding whether either represents a true wrist joint.

---

# 4. Human Hand Mate Hierarchy

The previously extracted hand hierarchy contains **43 mate features** across the main hand assembly and its nested finger/thumb assemblies.

| Assembly | Mates |
|---|---:|
| Main hand assembly | 15 |
| Index assembly | 6 |
| Middle assembly | 6 |
| Ring assembly | 6 |
| Pinky assembly | 6 |
| Thumb assembly | 4 |
| **Total** | **43** |

## 4.1 Main Hand Assembly

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

These establish the Palm-to-finger/thumb subassembly relationships.

## 4.2 Index Assembly

```text
02_Index/index_assem.SLDASM
```

Components:

```text
Index_Proximal
Index_Middle
Index_Distal
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

Intended structure:

```text
Palm
  |
  +-- Index Proximal
        |
        +-- MCP
             |
             +-- Index Middle
                   |
                   +-- PIP
                        |
                        +-- Index Distal
                              |
                              +-- DIP
```

The current assembly has explicit angular-limit mates for MCP and PIP. A dedicated DIP angular limit is not currently documented.

## 4.3 Middle Assembly

```text
03_Middle/Middle_Finger.SLDASM
```

Components:

```text
Middle_Proximal
Middle_Middle
Middle_Distal
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

## 4.4 Ring Assembly

```text
04_Ring/Ring_finger.SLDASM
```

Components:

```text
Ring_Proximal
Ring_Middle
Ring_Distal
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

## 4.5 Pinky Assembly

```text
05_Pinky/Pinky_finger.SLDASM
```

Components:

```text
Pinky_Proximal
Pinky_Middle
Pinky_Distal
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

## 4.6 Thumb Assembly

```text
06_Thumb/Thumb_finger.SLDASM
```

Components:

```text
Thumb_Base
Thumb_Proximal
Thumb_Distal_v2
```

Mates:

```text
Coincident2
Coincident3
LimitAngle1
Coincident4
```

Thumb CMC reference axes:

```text
Thumb_CMC_Axis_1
Thumb_CMC_Axis_2
```

A dummy/intermediate MuJoCo body can be used if both CMC rotational DOFs are retained.

---

# 5. Planned Human Hand MuJoCo Control

The physical simulation keeps five separate fingers, but the controller exposes **three high-level finger groups**.

### Group 1 — Index

Independent control:

```text
Index_MCP
Index_PIP
Index_DIP
```

### Group 2 — Middle + Ring + Pinky

The corresponding joint commands are coupled:

```text
middle_MCP = ring_MCP = pinky_MCP
middle_PIP = ring_PIP = pinky_PIP
middle_DIP = ring_DIP = pinky_DIP
```

Middle, Ring, and Pinky remain separate bodies so their different geometry and contact surfaces are preserved.

### Group 3 — Thumb

Independent thumb control:

```text
Thumb_CMC
Thumb_MCP
Thumb_IP
```

The CMC can be represented with one or two rotational DOFs depending on the final implementation.

---

# 6. Door CAD

Current door assembly:

```text
Door/Full_door_v3.SLDASM
```

Main components:

```text
base-1
frame-1
door-1
handle-1
```

The door model is based on the user's physical door, so the dimensions are intended to represent the real door.

---

# 7. Door Mate Hierarchy

Top-level Door assembly:

```text
Door/Full_door_v3.SLDASM
```

Extracted top-level mates:

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

Total top-level door mates: **10**.

The associated main components are:

```text
base-1
frame-1
door-1
handle-1
```

The exact component/reference association and exact values for `LimitAngle2` and `LimitAngle3` still need detailed mate-entity extraction before assigning final MuJoCo joint limits.

---

# 8. Door Reference Geometry

The SolidWorks door/frame assembly contains dedicated hinge and handle reference geometry.

Examples in the frame include:

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

Door-side hinge references include:

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

The handle contains:

```text
Handle center
Axis1
Axis2
```

These axes and points should be used to define MuJoCo joint frames instead of choosing arbitrary mesh coordinates.

---

# 9. Door Kinematic Intent

The intended structure is:

```text
Frame
  |
  +-- fixed
  |
  +-- Door
        |
        +-- hinge rotation
        |
        +-- Handle
```

The frame is fixed. The door rotates about the physical hinge axis. The handle is a separate component representing the push/pull mechanism.

---

# 10. Door Collision Requirements

The door was designed from the physical door, and the **door may geometrically intersect/overlap the frame in the CAD representation**.

For MuJoCo, this means door/frame mesh overlap should not automatically become a hard collision constraint that blocks normal hinge motion.

The handle has a separate requirement:

**The active handle must not intersect/pass through the frame.**

The two handle sides are oriented in opposite directions. One side is the side intended for the current interaction task; the opposite side is not being used for the current task.

Therefore the initial MuJoCo collision setup should focus on the active handle side and avoid unnecessary interaction/control complexity from the unused side.

---

# 11. Door Push/Pull Task

The active side of the door is intended to support both:

- pushing the door
- pulling the door

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

# 12. Planned MuJoCo Door Model

Conceptually:

```text
world
│
├── door_frame
│   └── fixed geometry
│
├── door
│   ├── door mesh
│   └── door_hinge
│
└── active_handle
    └── handle_joint
```

Whether the handle is an independently moving lever or a rigid part of the door can be finalized during the MuJoCo implementation.

---

# 13. Combined Hand + Door MuJoCo Scene

The intended scene is:

```text
world
├── forearm                    fixed
│    ↓
│   wrist
│    ↓
│   palm
│    ├── index
│    ├── middle
│    ├── ring
│    ├── pinky
│    └── thumb
│
└── door
     ├── frame
     ├── door hinge
     └── active handle
```

The high-level hand controls are:

```text
Control 1 → Index
Control 2 → Middle + Ring + Pinky
Control 3 → Thumb
```

The door provides its own hinge and handle dynamics.

---

# 14. CAD Mates vs MuJoCo Joints

The CAD mate system should not be copied one-to-one into MuJoCo.

SolidWorks mates describe assembly relationships such as positioning, alignment, orientation, and limits. MuJoCo requires an explicit physics model consisting of bodies, joints, axes, limits, meshes, collision pairs, contacts, and actuators.

The CAD mates therefore act as the source of the intended mechanical relationships, while the MuJoCo model is built specifically for simulation.

---

# 15. Current Project Status

## Human Hand

- [x] Palm modeled
- [x] MCP reference points created
- [x] Wrist reference created
- [x] Thumb CMC reference created
- [x] Thumb CMC two-axis reference structure created
- [x] Index geometry modeled
- [x] Middle geometry modeled
- [x] Ring geometry modeled
- [x] Pinky geometry modeled
- [x] Thumb geometry modeled
- [x] Four fingers assembled
- [x] Thumb assembled
- [x] Main hand assembly created
- [x] Hand mate hierarchy extracted
- [x] 43 hand mates identified
- [x] Three-group MuJoCo control concept defined

## Wrist / Forearm

- [x] Wrist added
- [x] Forearm added as support structure
- [x] Integrated `Hand_forearm.SLDASM` assembly created
- [x] Top-level integrated mate tree extracted
- [x] 5 top-level integrated assembly mates identified
- [ ] Verify detailed entity data for `LimitAngle3` and `LimitAngle5`
- [ ] Finalize whether any wrist DOF is required

## Door

- [x] Door CAD modeled
- [x] Door based on physical door dimensions
- [x] Frame modeled
- [x] Door modeled
- [x] Handle modeled
- [x] Hinge reference geometry created
- [x] Handle reference geometry created
- [x] Door mate hierarchy extracted
- [x] 10 top-level door mates identified
- [x] Active handle side identified
- [x] Opposite handle side excluded from the current interaction task
- [x] Door/frame collision behavior defined conceptually
- [ ] Extract detailed door mate entities and exact angle limits
- [ ] Determine exact hinge axis from mate/reference data
- [ ] Determine exact handle joint axis and limits
- [ ] Build MuJoCo door model
- [ ] Build combined hand + door scene
- [ ] Validate contacts and collisions

---

# 16. Mate-Tree Extraction Notes

The latest integrated extraction was run on:

```text
Hand_forearm.SLDASM
```

The first section of the report is the actual top-level integrated assembly. It lists the five integrated mates and then the direct components `wrist-1` and `human hand v3-2`. fileciteturn1file0L9-L32

The extraction then recursively enters `human hand v3.SLDASM` and its nested Index, Pinky, Ring, Thumb, and Middle assemblies. The raw recursive report can therefore show the same nested assembly more than once because the traversal is enumerating component trees recursively. The repeated text does **not** mean extra physical fingers or extra physical joints exist.

---

# 17. MuJoCo Starting Point

The CAD phase is now ready to transition toward MuJoCo.

The first simulation should prioritize:

1. Correct mesh placement and scale.
2. Correct body hierarchy.
3. Correct joint axes.
4. Correct hinge/handle behavior.
5. Correct hand-to-handle contact.
6. Correct door/frame collision filtering.
7. Three high-level hand control groups.

Target starting structure:

```text
world
│
├── forearm                   fixed
│    ↓
│   wrist
│    ↓
│   palm
│    ├── index
│    ├── middle
│    ├── ring
│    ├── pinky
│    └── thumb
│
└── door
     ├── frame
     ├── door_hinge
     └── active_handle
```

The next concrete step is to extract the **detailed mate entities/axes/limits** for the integrated forearm assembly and the door, then use those values to define the MuJoCo joint frames before final mesh import.
