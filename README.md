# Human Hand + Door — CAD and MuJoCo Documentation

## 1. Project Overview

This repository contains the SolidWorks CAD models and planned MuJoCo simulation structure for a human hand interacting with a door.

The project currently contains two major CAD systems:

1. Human hand
2. Door assembly

The long-term goal is to port both systems into MuJoCo while preserving the important kinematic relationships from SolidWorks and using physically meaningful collision/contact behavior.

---

# 2. Repository Structure

```text
Human_Hand_200mm/
├── 00_Master/
├── 01_Palm/
├── 02_Index/
├── 03_Middle/
├── 04_Ring/
├── 05_Pinky/
├── 06_Thumb/
├── 07_Assembly/
├── Door/
├── Macro/
├── README.md
├── human hand v1.SLDASM
├── human hand v2.SLDASM
└── human hand v3.SLDASM
```

Current door assembly:

```text
Door/Full_door_v3.SLDASM
```

---

# 3. Human Hand CAD

The physical CAD model contains five separate fingers:

```text
Palm
├── Index
├── Middle
├── Ring
├── Pinky
└── Thumb
```

The planned MuJoCo controller uses **three independent control groups**:

```text
Control Group 1 → Index
Control Group 2 → Middle + Ring + Pinky
Control Group 3 → Thumb
```

The three physical fingers in Group 2 remain separate bodies in the simulation, but corresponding joint commands are coupled.

---

# 4. Human Hand Mate Hierarchy

The extracted SolidWorks mate hierarchy contains **43 mate features**:

| Assembly | Mates |
|---|---:|
| Main hand assembly | 15 |
| Index assembly | 6 |
| Middle assembly | 6 |
| Ring assembly | 6 |
| Pinky assembly | 6 |
| Thumb assembly | 4 |
| **Total** | **43** |

## Main hand assembly

`human hand v3.SLDASM`

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

## Index

`02_Index/index_assem.SLDASM`

```text
Coincident1
Coincident2
LimitAngle2
Coincident4
LimitAngle5
Coincident6
```

Components:

```text
Index_Proximal
Index_Middle
Index_Distal
```

## Middle

`03_Middle/Middle_Finger.SLDASM`

```text
Coincident11
Coincident13
LimitAngle2
Coincident15
LimitAngle4
Coincident18
```

Components:

```text
Middle_Proximal
Middle_Middle
Middle_Distal
```

## Ring

`04_Ring/Ring_finger.SLDASM`

```text
Coincident1
Coincident2
LimitAngle1
Coincident3
Coincident4
LimitAngle2
```

Components:

```text
Ring_Proximal
Ring_Middle
Ring_Distal
```

## Pinky

`05_Pinky/Pinky_finger.SLDASM`

```text
Coincident1
Coincident2
LimitAngle1
Coincident3
Coincident4
LimitAngle2
```

Components:

```text
Pinky_Proximal
Pinky_Middle
Pinky_Distal
```

## Thumb

`06_Thumb/Thumb_finger.SLDASM`

```text
Coincident2
Coincident3
LimitAngle1
Coincident4
```

Components:

```text
Thumb_Base
Thumb_Proximal
Thumb_Distal_v2
```

The thumb also has two CMC reference axes:

```text
Thumb_CMC_Axis_1
Thumb_CMC_Axis_2
```

A dummy/intermediate body can be used in MuJoCo if two independent CMC rotational DOFs are retained.

---

# 5. Planned Human Hand MuJoCo Control

### Group 1 — Index

Independent:

```text
Index_MCP
Index_PIP
Index_DIP
```

### Group 2 — Middle + Ring + Pinky

Corresponding commands are shared:

```text
middle_MCP = ring_MCP = pinky_MCP
middle_PIP = ring_PIP = pinky_PIP
middle_DIP = ring_DIP = pinky_DIP
```

The physical fingers remain separate.

### Group 3 — Thumb

```text
Thumb_CMC
Thumb_MCP
Thumb_IP
```

The CMC may use two rotational DOFs depending on the final MuJoCo implementation.

---

# 6. Door CAD

Current assembly:

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

The door was modeled from the user's physical door, so the current CAD dimensions are intended to represent the physical door.

The SolidWorks tree contains hinge and handle reference geometry, including:

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

These references should be used when defining the MuJoCo joint frames.

---

# 7. Door Mate Hierarchy

The top-level Door assembly was inspected using the same SolidWorks mate-tree extraction method.

Assembly:

```text
Door/Full_door_v3.SLDASM
```

Mate group:

```text
Mates
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

**Total top-level door mates: 10**

The associated assembly components are:

```text
base-1
frame-1
door-1
handle-1
```

The exact component/reference association and exact limits for `LimitAngle2` and `LimitAngle3` still need detailed mate-entity extraction before assigning their final MuJoCo meaning.

---

# 8. Door Kinematic Intent

The intended physical structure is:

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

The frame is fixed.

The door rotates around the physical hinge axis.

The handle is a separate component and represents the push/pull mechanism.

---

# 9. Door Collision Requirements

The CAD model is based on the physical door, and **door/frame geometric intersection is acceptable for the current CAD representation**.

Therefore, in MuJoCo, the door and frame should not automatically be treated as a hard collision pair everywhere. Collision geometry should instead be selected/configured to permit the intended hinge motion.

The handle is different:

**The active handle must not intersect/pass through the frame.**

The handle geometry on the two sides is oriented in opposite directions. One side functions as the relevant locking/handle side for the current task.

For the current simulation, **only one side of the door is being used for interaction**. The opposite side does not need to be an independently controlled interaction mechanism in the first MuJoCo version.

---

# 10. Door Push/Pull Task

The active side of the door is intended to support both:

- pushing the door
- pulling the door

The interaction chain is:

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

The unused opposite-side handle should not introduce unnecessary collision/control complexity in the first simulation.

---

# 11. Planned MuJoCo Door Structure

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
└── handle
    └── handle_joint
```

The exact body nesting can be changed depending on whether the handle is treated as a separate actuated lever or as a rigid part of the door.

Because the SolidWorks model contains a separate `handle` component, retaining it as a separate MuJoCo body is useful if handle motion is required.

---

# 12. Hand + Door MuJoCo Scene

The final simulation is intended to combine both systems:

```text
                         HAND
                           |
                           v
                    +-------------+
                    |   HANDLE    |
                    +-------------+
                           |
                           v
                    +-------------+
                    |    DOOR     |
                    +-------------+
                           ||
                           || hinge
                           ||
                    +-------------+
                    |    FRAME    |
                    +-------------+
```

Hand control:

```text
Index
Middle/Ring/Pinky
Thumb
```

Door dynamics:

```text
Door Hinge
Active Handle
```

---

# 13. Collision Strategy

Collision should be designed separately from the SolidWorks mate system.

Important intended contacts:

```text
Hand ↔ Handle        ENABLE
Hand ↔ Door          ENABLE where relevant
Hand ↔ Frame         ENABLE only where physically relevant

Door ↔ Frame         selectively handled
Handle ↔ Frame       must not pass through frame
```

The CAD overlap between the door and frame should not automatically become a simulation collision constraint.

---

# 14. CAD Mates vs MuJoCo Joints

SolidWorks mates should be treated as the source of the mechanical relationships, not as a one-to-one list of MuJoCo joints.

SolidWorks provides:

- positioning
- alignment
- orientation
- limits
- assembly relationships

MuJoCo requires explicit:

- bodies
- joint types
- joint positions
- joint axes
- joint limits
- collision geometries
- actuators
- contact settings

Therefore the final MuJoCo model will be constructed from the CAD kinematic information rather than mechanically copying every CAD mate.

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

## Door

- [x] Door CAD modeled
- [x] Door dimensions based on the physical door
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

# 16. Next Steps

## Step 1 — Detailed Door Mate Extraction

Extract for all 10 door mates:

```text
Mate name
Mate type
Component 1
Component 2
Reference type
Reference geometry
Position
Direction
Limits
Alignment
```

This will determine exactly which `LimitAngle` controls the hinge and which controls the handle mechanism.

## Step 2 — Finalize Door Joints

Create:

```text
door_hinge
handle_joint
```

with the correct axes and limits.

## Step 3 — Export Geometry

Prepare separate MuJoCo meshes for:

```text
Palm
Index Proximal
Index Middle
Index Distal
Middle Proximal
Middle Middle
Middle Distal
Ring Proximal
Ring Middle
Ring Distal
Pinky Proximal
Pinky Middle
Pinky Distal
Thumb Base
Thumb Proximal
Thumb Distal

Door Frame
Door
Active Handle
```

The unused handle side can be excluded from active interaction or retained as visual geometry depending on the final scene.

## Step 4 — Build MuJoCo Kinematic Tree

Construct the hand and door as separate articulated systems in the same world.

## Step 5 — Add Actuators

Hand:

```text
Index controls
Middle/Ring/Pinky shared controls
Thumb controls
```

Door:

```text
Door hinge
Handle/lever
```

## Step 6 — Add Contact

Tune contact so that:

- the hand can physically interact with the active handle
- the handle cannot pass through the frame
- the door can rotate normally around its hinge
- irrelevant door/frame CAD overlap does not prevent the desired motion

---

# 17. Final High-Level System

```text
                    MUJOCO WORLD
                         |
          +--------------+--------------+
          |                             |
         HAND                          DOOR
          |                             |
    +-----+------+                +-----+------+
    |     |      |                |            |
  Index  M/R/P  Thumb           Hinge       Handle
```

Where:

```text
M/R/P = Middle + Ring + Pinky
```

The hand has **three high-level control groups**, while the door is a separate articulated object with a hinge and active handle interaction.

The SolidWorks CAD remains the source geometry and mechanical reference, while MuJoCo provides the final physics, contacts, joints, and control system.
