# Human Hand + Door — SolidWorks, sw2robot, and MuJoCo Documentation

> **Project status (2026-09-28):** **Hand** — `Mujoko/Hand_forearm_v5`: 17 CAD joints + thumb MCP sideways hinge = **18 position actuators**, 2-DOF ball-centre wrist, thumb MCP universal joint, every finger/thumb joint 0° = straight → 90° = bent, gamepad / sliders / saved poses / keyboard arm control (§8). **Door** — `Mujoko/Full_door_v4`: passive door (hinge + handle) built from `Full_door_v4.SLDASM`, upright, correct masses, handle return spring, handle ↔ frame solid collision, scripted scene and **manual keyboard control** (§37). Next: put the hand in front of the door (§37.9).
>
> **Pipeline:** SolidWorks → sw2robot (extract + build) → `drop_angle_mates.py` / `fix_axes.py` → MJCF → `add_actuators.py` → `hand_controller.py`. Full commands in §35.3.
>
> **Source of truth:** SolidWorks CAD defines the intended geometry, assembly relationships, reference geometry, and joint limits. MuJoCo is used for simulation physics, contacts, actuators, and control.
## Team
Capstone project, B.Tech CSE, Shiv Nadar University Chennai
- @RawMangoking
- @narenkumarchandran
- @plan28-06

Mentor: Dr. Priya GL
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

Current CAD and simulation organization:

```text
Human_Hand_200mm/
├── 00_Master/
│   ├── Hand_Master_200mm.SLDPRT
│   └── Hand_Parameters.txt
├── 01_Palm/
│   └── Palm.SLDPRT
├── 02_Index/        Index_Proximal / Index_Middle / Index_Distal .SLDPRT, index_assem.SLDASM
├── 03_Middle/       Middle_Proximal / Middle_Middle / Middle_Distal .SLDPRT, Middle_Finger.SLDASM
├── 04_Ring/         Ring_Proximal / Ring_Middle / Ring_Distal .SLDPRT, Ring_finger.SLDASM
├── 05_Pinky/        Pinky_Proximal / Pinky_Middle / Pinky_Distal .SLDPRT, Pinky_finger.SLDASM
├── 06_Thumb/        Thumb_Base / Thumb_Proximal / Thumb_Distal_v2 .SLDPRT, Thumb_finger.SLDASM
├── 07_Assembly/
│   └── Human_Hand.SLDASM
├── Door/
│   ├── Full_door_v3.SLDASM          (previous)
│   └── Full_door_v4.SLDASM          ← current door: base, frame, door, handle
├── Macro/
├── Hand_forearm.SLDASM              ← integrated forearm + wrist ball + hand (top assembly)
├── human hand v3.SLDASM             ← hand assembly used by Hand_forearm
├── sw2+mujoko+python/               ← export / simulation tools (§36)
│   ├── drop_angle_mates.py          hand + door: remove LimitAngle records from graph.json
│   ├── fix_axes.py                  hand + door: joint axes from the mates
│   ├── add_actuators.py             hand: actuators, anatomical zero, upright
│   ├── hand_controller.py           hand: gamepad / keyboard / poses
│   ├── check_pose.py                hand: part-by-part pose comparison
│   ├── door_setup.py                door: upright, masses, passive joints, handle-frame collision, checks
│   ├── door_scene.py                door: scripted push/turn/open scene + manual keyboard control
│   ├── door_env.py                  RL: Gymnasium env, hand + door, tasks reach / handle / door (§38)
│   ├── train_policy.py              RL: SAC training / playback with stable-baselines3
│   ├── env_tests.py                 RL: environment test battery (PASS/FAIL report)
│   ├── Hand_forearm.joints.yaml     ← repo copy of the hand's sw2robot joint config
│   ├── Full_door_v4.joints.yaml     ← repo copy of the door's sw2robot joint config
│   └── output/                      ← sw2robot packages (generated, in .gitignore)
├── Mujoko/
│   ├── Hand_forearm_v5/             ← CURRENT hand model (18 actuators; same layout as v4)
│   ├── Full_door_v4/                ← CURRENT door model
│   │   └── mjcf/  Full_door_v4.xml (export), Full_door_v4_sim.xml (use this), door_config.json, assets/
│   ├── Hand_forearm_v4/             ← previous hand model (18 actuators)
│   │   ├── mjcf/
│   │   │   ├── Hand_forearm.xml             sw2robot export (untouched)
│   │   │   ├── Hand_forearm_actuated.xml    add_actuators.py output — use this one
│   │   │   └── hand_config.json             joint groups / open-closed poses for the controller
│   │   └── assets/                          STL meshes
│   ├── Hand_forearm_mjcf _v3/       ← old 15-joint hand (reference)
│   └── Hand_Door/                   ← combined scene (old hand; see §35.8)
└── README.md
```

The live sw2robot package (graph.json, meshes, joints yaml) is generated in the Windows temp folder and can be wiped by Windows at any time:

```text
C:\Users\naren\AppData\Local\Temp\sw2robot\output\Hand_forearm        package (graph.json, urdf, joints.yaml)
C:\Users\naren\AppData\Local\Temp\sw2robot\output\Hand_forearm_mjcf   MuJoCo package written by the build
```

Always copy the MJCF into `Mujoko/Hand_forearm_v4/` and the yaml into `sw2+mujoko+python/` after a successful build.

The project also contains earlier hand assembly versions such as `human hand v1.SLDASM`, `human hand v2.SLDASM`, and `human hand v3.SLDASM`.

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

The recursive mate-tree macro (latest run 2026-09-27) finds **52 mate features**: 6 in the top-level `Hand_forearm` assembly (§6) and **46** in `human hand v3` and its five nested finger/thumb sub-assemblies.

| Assembly | Mates | Joints it defines |
|---|---:|---|
| `Hand_forearm` (top, §6) | 6 | wrist tilt, wrist flexion |
| `human hand v3` (main) | 16 | index / middle / ring / pinky MCP, thumb CMC |
| Index | 6 | PIP, DIP |
| Middle | 6 | PIP, DIP |
| Ring | 6 | PIP, DIP |
| Pinky | 6 | PIP, DIP |
| Thumb | 6 | MCP (flexion + lateral), IP |
| **Total** | **52** | **18 motions** |

Mate types in the macro output: `MateCoincident`, `MateLimitPlanarAngleDim` (LimitAngle), `MatePerpendicular`.

**Hinge pattern used everywhere:** one *point-to-point* Coincident at the joint centre + one *axis-to-axis* Coincident along the hinge + one LimitAngle for the range. sw2robot derives the hinge axis from the axis-to-axis Coincident (see §35.2 for the LimitAngle caveat).

## 5.1 Main hand assembly

Assembly: `human hand v3.SLDASM`

Components: `Palm-1`, `index_assem-2`, `Pinky_finger-1`, `Thumb_finger-1`, `Ring_finger-1`, `Middle_Finger-1`

Mates (16):

```text
Coincident2   Coincident3   Coincident4   Coincident5   Coincident7
LimitAngle1   Coincident9   LimitAngle2   Coincident11  LimitAngle3
Coincident13  LimitAngle4   Coincident14  LimitAngle10  Coincident17
Perpendicular1
```

Interpretation: 10 Coincidents = 2 per digit (point + axis), 5 LimitAngles = 1 per digit (MCP / CMC range), `Perpendicular1` belongs to the thumb CMC. (The earlier README listed 15 mates — `Perpendicular1` was missing.)

## 5.2 Index assembly — `02_Index/index_assem.SLDASM`

Components: `Index_Proximal-1`, `Index_Middle-1`, `Index_Distal-1`

```text
Coincident1  Coincident2  LimitAngle2  Coincident4  LimitAngle5  Coincident6
```

Two hinges (PIP, DIP), each 2 Coincidents + 1 LimitAngle. The index MCP (palm → proximal) is defined in the main assembly.

```text
Palm → Index Proximal (MCP, main assembly) → Index Middle (PIP) → Index Distal (DIP)
```

## 5.3 Middle assembly — `03_Middle/Middle_Finger.SLDASM`

Components: `Middle_Proximal-5`, `Middle_Middle-4`, `Middle_Distal-4`

```text
Coincident11  Coincident13  LimitAngle2  Coincident15  LimitAngle3  Coincident18
```

(Corrected: the earlier README listed `LimitAngle4`; the macro reports `LimitAngle3`.)

## 5.4 Ring assembly — `04_Ring/Ring_finger.SLDASM`

Components: `Ring_Proximal-1`, `Ring_Middle-1`, `Ring_Distal-1`

```text
Coincident1  Coincident2  LimitAngle1  Coincident3  Coincident4  LimitAngle2
```

## 5.5 Pinky assembly — `05_Pinky/Pinky_finger.SLDASM`

Components: `Pinky_Proximal-1`, `Pinky_Middle-1`, `Pinky_Distal-1`

```text
Coincident1  Coincident2  LimitAngle1  Coincident3  Coincident4  LimitAngle2
```

Note: sw2robot could not turn the ring/pinky mates into hinge axes automatically (it reports "under-constrained"); the axes are taken from the mated reference axes by `fix_axes.py` instead (§35.2).

## 5.6 Thumb assembly — `06_Thumb/Thumb_finger.SLDASM`

Components: `Thumb_Base-1`, `Thumb_Proximal-1`, `Thumb_Distal_v2-1`

Mates (6 — the earlier README listed 4):

```text
Coincident2  LimitAngle3  LimitAngle1  Coincident4  Coincident5  LimitAngle2
```

Interpretation (confirmed from the sw2robot graph, which recorded two LimitAngles on Base↔Proximal and one on Proximal↔Distal):

```text
Thumb_Base ↔ Thumb_Proximal   (MCP)  point-only Coincident + 2 LimitAngles
                                     → flexion + LATERAL (left/right) motion, 2 DOF
Thumb_Proximal ↔ Thumb_Distal (IP)   point + axis Coincident + 1 LimitAngle
```

Because the MCP is mated point-only, its flexion axis is inferred (parallel to the IP axis) and the **lateral axis is added in MuJoCo** with `--add-axis "Thumb_Proximal_1@LO,HI"` at the same pivot point (§35.5).

Thumb CMC references in the palm: `Thumb_CMC_Axis_1`, `Thumb_CMC_Axis_2` (available if a 2-DOF CMC is wanted later).

# 6. Integrated Hand + Wrist + Forearm CAD

Integrated assembly: `Hand_forearm.SLDASM`

Top-level components:

```text
wrist-1            (the wrist ball)
fore arm-1         (fixed base)
human hand v3-2
```

Top-level mates (6, after the 2026-09-27 wrist rebuild):

```text
Coincident1  Coincident2  LimitAngle1  Coincident3  Coincident4  LimitAngle3
```

= two hinges, each *centre-point Coincident + axis Coincident + LimitAngle*.

## 6.1 Ball-centre wrist (2 DOF)

The wrist ball is the middle piece of a universal joint. **Both wrist axes pass through the ball centre**, so the palm rolls around the ball and never separates from it.

```text
Forearm ──(TILT hinge, ±10°)──► Wrist ball ──(FLEXION hinge)──► Palm
             both axes pass through the ball centre, perpendicular to each other
```

Reference geometry (each created inside its own part with *Open Part*, not in the assembly):

| Part | Point | Axis |
|---|---|---|
| wrist ball | `center` (sphere centre) | `flex_axis` = `center` + Front Plane, `tilt_axis` = `center` + Right Plane |
| fore arm | `ballcenter` = (plane ⟂ long axis through `wrist point`, offset by ball radius r) ∩ long axis | `tilt_axis` through `ballcenter` |
| palm | `ballcenter` (same construction, offset toward the forearm) | `flex_axis` through `ballcenter`, normal to a plane ⟂ `MCP_REFERENCE` → exactly parallel to the knuckles |

Mates:

```text
fore arm ↔ ball :  ballcenter ↔ center,  tilt_axis ↔ tilt_axis,  LimitAngle −10° … +10°
ball ↔ palm     :  center ↔ ballcenter,  flex_axis ↔ flex_axis,  LimitAngle (flexion range)
```

Verified with *Measure*: both `ballcenter` points report the same XYZ as the ball `center`.

## 6.2 Why the wrist was rebuilt

The previous top-level mates (`Coincident5`, `Coincident6`, `Coincident7`, `LimitAngle3`, `LimitAngle5`, now deleted) attached the palm to `wrist up` and the forearm to `wrist down` — points on the ball **surface**. The palm therefore rotated about a point on the surface: one side opened a gap and the other sank into the ball ("dislocation"). `Coincident7` was also point-only, which let the palm sit twisted ~8° about the forearm axis. The old forearm↔wrist rotation about the forearm's long axis (twist) was dropped; pronation/supination is a forearm motion and is not modelled.

## 6.3 SolidWorks drag behaviour

Dragging the palm through a **large angle in one move** can still make it look dislocated in SolidWorks, while small steps (0° → 5° → 10°) do not. That is the SolidWorks mate solver jumping to a different solution during a big drag (limit mates are known to be solved loosely while dragging), not a model error. For CAD checks, move in small steps or set the angle with a temporary Angle mate. MuJoCo is not affected: its joints are exact hinges, so the palm cannot separate from the ball, and `hand_controller.py` already limits how fast any joint target moves (`--max-rate`, §36.4).

# 7. Human Hand MuJoCo Export (v4, 2026-09-27)

Current model: `Mujoko/Hand_forearm_v4/mjcf/Hand_forearm_actuated.xml`

```text
18 links
17 revolute joints from CAD   (2 wrist + 3 thumb + 4 × 3 fingers)
+1 hinge added in MuJoCo       (thumb MCP lateral)
18 position actuators
forearm welded to the world (fixed base), model rotated so forearm → palm points up (+Z)
```

Kinematic tree:

```text
world
└── fore_arm_1                              (fixed)
    └── wrist_1                             wrist TILT        (CAD axis, ball centre)
        └── human_hand_v3_2__Palm_1         wrist FLEXION     (CAD axis, ball centre)
            ├── …Thumb_Base_1               thumb CMC         (CAD)
            │   └── …Thumb_Proximal_1       thumb MCP flexion (inferred) + MCP LATERAL (added)
            │       └── …Thumb_Distal_v2_1  thumb IP          (CAD)
            ├── …Index_Proximal_1 → …Index_Middle_1 → …Index_Distal_1        MCP, PIP, DIP (CAD)
            ├── …Middle_Proximal_5 → …Middle_Middle_4 → …Middle_Distal_4     MCP, PIP, DIP (CAD)
            ├── …Ring_Proximal_1 → …Ring_Middle_1 → …Ring_Distal_1           MCP, PIP, DIP (fix_axes)
            └── …Pinky_Proximal_1 → …Pinky_Middle_1 → …Pinky_Distal_1        MCP, PIP, DIP (fix_axes)
```

The earlier v3 export (15 hinge joints, sw2robot's default `kp = 50` actuators) is superseded. How the model is produced: §35.3. Actuator list: §35.5.

## 7.1 Degrees of freedom per joint

Every joint is a single-axis **hinge (revolute, 1 DOF)**. The thumb MCP carries **two** hinges on the same pivot (a universal joint), so it has 2 DOF. Angles: 0° = straight / neutral, + = toward the palm (flexion).

| Joint | Anatomical name | Between (parent → child) | Type | DOF | Motion | Range | Axis from |
|---|---|---|---|:-:|---|---|---|
| Wrist tilt | radial / ulnar deviation | fore_arm → wrist ball | hinge | 1 | side-to-side | −10° … +10° | CAD (ball centre) |
| Wrist flex | flexion / extension | wrist ball → palm | hinge | 1 | forward / back | −57° … +57°* | CAD (ball centre) |
| Thumb CMC | carpometacarpal | palm → Thumb_Base | hinge | 1 | thumb out of the palm plane | 0° (in palm plane) … 90° | CAD |
| Thumb MCP bend | metacarpophalangeal flexion | Thumb_Base → Thumb_Proximal | hinge | 1 | bend | 0° … 90° | Perpendicular mate (base axis), squared to bone |
| Thumb MCP sideways | MCP abduction / adduction | Thumb_Base → Thumb_Proximal (2nd hinge) | hinge | 1 | sideways | −45° … +45° | added in MuJoCo (`--add-axis`) |
| Thumb IP | interphalangeal | Thumb_Proximal → Thumb_Distal | hinge | 1 | bend | 0° … 90° | CAD |
| Index MCP / PIP / DIP | knuckle / middle / tip joint | palm → proximal → middle → distal | hinge ×3 | 3 | bend | 0° … 90° each | CAD |
| Middle MCP / PIP / DIP | | palm → proximal → middle → distal | hinge ×3 | 3 | bend | 0° … 90° each | CAD |
| Ring MCP / PIP / DIP | | palm → proximal → middle → distal | hinge ×3 | 3 | bend | 0° … 90° each | CAD reference axes (`fix_axes.py`) |
| Pinky MCP / PIP / DIP | | palm → proximal → middle → distal | hinge ×3 | 3 | bend | 0° … 90° each | CAD reference axes (`fix_axes.py`) |

\* placeholder until the wrist-flexion LimitAngle values are entered in the yaml.

**DOF totals**

| Group | DOF | Joints |
|---|:-:|---|
| Wrist | 2 | tilt, flexion |
| Thumb | 4 | CMC, MCP bend, MCP sideways, IP |
| Index | 3 | MCP, PIP, DIP |
| Middle | 3 | MCP, PIP, DIP |
| Ring | 3 | MCP, PIP, DIP |
| Pinky | 3 | MCP, PIP, DIP |
| **Hand + wrist (actuated)** | **18** | 18 position actuators, one per DOF |
| Moving base in keyboard mode | +6 | free joint of the forearm (x, y, z, 3 rotations), moved by the keyboard target, not actuated |

For comparison, a human hand is usually modelled with 21–27 DOF: the extra ones are finger MCP sideways spread (4), a second thumb-CMC axis, and forearm twist. They can be added later with `--add-axis` (e.g. `--add-axis "Index_Proximal_1@-15,15"` for index spread).

# 8. Human Hand MuJoCo Control Design

## 8.1 Planned high-level groups

The physical simulation keeps the five fingers separate; the planned high-level controller exposes three groups:

```text
Group 1 → Index                    Index MCP / PIP / DIP
Group 2 → Middle + Ring + Pinky    corresponding MCP / PIP / DIP commands shared
Group 3 → Thumb                    CMC, MCP flexion, MCP lateral, IP
```

## 8.2 Implemented gamepad controller — `hand_controller.py`

Xbox layout (PlayStation / Switch pads are remapped automatically by SDL).

**SYNERGY mode (default)**

| Input | Action |
|---|---|
| Right trigger | close index + middle + ring + pinky (one "curl" per finger drives MCP, PIP, DIP) |
| Left trigger | close thumb (CMC, MCP, IP) |
| Left stick up/down | wrist flexion (rate control — stays where you leave it) |
| Left stick left/right | wrist tilt ±10° |
| Right stick left/right | thumb MCP lateral (all `--add-axis` joints) |
| A / B / X / Y | preset: open / fist / pinch / point |
| RB / LB | hold current grip / release hold |
| Right-stick click | centre the wrist |

**JOINT mode** (Start toggles): D-pad left/right selects a group (wrist, thumb, index, middle, ring, pinky, spread), D-pad up/down selects a joint, right stick up/down moves it. **Back** resets everything.

Without a gamepad the viewer's *Control* panel gives one slider per actuator.

The 3-group coupling of §8.1 (index separate from middle+ring+pinky) is not yet mapped to the gamepad — currently the right trigger drives all four fingers together. Next step in §35.8.

## 8.3 Keyboard mode — move the whole arm (`--keyboard`)

```powershell
python hand_controller.py "<model>\mjcf\Hand_forearm_actuated.xml" --keyboard
```

A small **"Hand keyboard control"** window opens next to the viewer — **click it first**; the keys go to that window (inside the MuJoCo viewer W/S/A/D/P are display shortcuts).

| Key | Action |
|---|---|
| W / S | move arm forward / back (world +Y / −Y) |
| A / D | move arm left / right (−X / +X) |
| + / − (also PgUp / PgDn) | move arm up / down |
| Num8 / Num2 (↑ / ↓) | tilt forward / back |
| Num4 / Num6 (← / →) | tilt left / right |
| Num7 / Num9 (Q / E) | twist about the arm's own axis |
| Shift / Ctrl (hold) | 4× faster / 4× finer |
| R | reset the arm to its start position |
| P | bring up the saved pose (P again → open hand) |
| 0 | open hand |
| 1 – 9 | saved poses in the order they were saved |
| N | save what the hand shows now (e.g. after using the sliders) as a new pose `pose_<n>` |
| Esc | quit |

How it works: the forearm gets a free joint and is held by a **soft weld** to an invisible target (small green sphere) that the keys move, so the arm still collides and pushes physically (important for the door). Tilt and twist pivot about the **wrist ball** (`--pivot base` to use the forearm origin). A 5 cm **leash** keeps the target from running away when the arm is blocked (floor, door). The model files are not changed — the movable version is built in memory at start. Options: `--p-pose NAME` (pose for P; default = last saved), `--move-speed` (m/s, 0.15), `--turn-speed` (deg/s, 60), `--fixed-base` (poses only, arm fixed), `--load-pose NAME` (start in that pose). The Control sliders stay usable between pose recalls.

## 8.4 Saved poses

Poses are stored **by joint name** in `poses.json` next to the model, so they survive rebuilds (copy `poses.json` into a new model folder to keep them).

```powershell
python hand_controller.py <model> --save-pose NAME                  # set sliders, CLOSE the window -> saved
python hand_controller.py <model> --save-pose NAME --ctrl "v1,v2,…"  # from Control-panel values (radians, panel order)
python hand_controller.py <model> --load-pose NAME                  # hold it
python hand_controller.py <model> --list-poses
```

Re-using a name overwrites that pose.

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

> **2026-09-27:** these counts are for the old 15-joint hand. With the v4 hand (18 actuators) the combined scene will have 18 + 2 = 20 actuators — see §35.8.

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

## Hand (v4, generated by `add_actuators.py`)

sw2robot's actuators are replaced by one **position actuator per hinge**, sized from the load each joint actually carries (important for a 200 mm hand whose distal links weigh a few grams):

```text
τ_g       = subtree mass × g × lever arm (joint axis → subtree centre of mass, open pose)
kp        = max(0.02, τ_g / 0.02 rad)        → droops ≤ ~0.02 rad under gravity
armature  = estimated inertia about the axis  (numerical stability at 2 ms)
damping   = 2 · ζ · sqrt(kp · (I + armature)),  ζ = 1 (critical)
forcerange= ± 0.5 rad × kp
ctrlrange = joint range
integrator= implicitfast
```

Tuning flags: `--kp` (fixed gain for all), `--sag`, `--kp-min`, `--zeta`, `--force-margin`. The per-joint values are printed by `add_actuators.py` and stored in `hand_config.json`.

Contacts that already touch in the open pose (CAD clearances ≈ 0) are excluded automatically with `<contact><exclude>`; `--keep-contacts` disables this.

## Door (unchanged)

```xml
<position ... kp="50" ... forcerange="-10 10" />
```

Door joint damping: `3.18471338`. These are starting values to tune later for contact stability.

# 24. SolidWorks Macro Work Completed

## Macro 1 — Recursive hand/forearm mate tree

Purpose: traverse the MateGroup recursively, identify nested sub-assemblies, list every mate and mate type, list the component hierarchy.

Latest result (2026-09-27, after the wrist rebuild):

```text
Hand_forearm.SLDASM                 6 top-level mates
└── human hand v3.SLDASM           16 main-hand mates
    ├── index_assem                 6
    ├── Pinky_finger                6
    ├── Thumb_finger                6
    ├── Ring_finger                 6
    └── Middle_Finger               6
                                   ──
                                   52 mates total
```

Note: the macro prints each sub-assembly twice (once while recursing, once from the flattened component list) — the counts above are unique mates.

## Macro 2 — Detailed door mate/entity extraction

Purpose: find the MateGroup, read each `Mate2`, `GetMateEntityCount`, each `MateEntity`, `ReferenceComponent`, `ReferenceType`, `EntityParams`. This macro established the geometry vectors used to diagnose the door hinge and handle axes. The same macro can be run on `Hand_forearm` to print every hand mate's points and axes if a joint needs checking.

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

- [x] Master hand planning, palm, fingers, thumb modelled
- [x] Palm reference geometry, MCP points, wrist reference, thumb CMC point + two CMC axes
- [x] Finger sub-assemblies and main hand assembly
- [x] Wrist/forearm assembly
- [x] **Wrist rebuilt as 2-DOF ball-centre joint (tilt ±10° + flexion)**
- [x] Recursive mate tree extracted: 6 top-level + 46 hand mates = 52
- [x] sw2robot: all five finger sub-assemblies expanded (`expand:` fixed)
- [x] sw2robot: LimitAngle-lock worked around (`drop_angle_mates.py`)
- [x] sw2robot: all 17 CAD joints have real axes (`fix_axes.py` for ring, pinky, thumb MCP)
- [x] MJCF build working (absolute package path)
- [x] 18 position actuators with load-based gains (`add_actuators.py`)
- [x] Thumb MCP lateral axis added in MuJoCo
- [x] Model stands upright in MuJoCo (`--point-up`)
- [x] Gamepad controller + joint sweep (`hand_controller.py`)
- [x] Saved poses, keyboard control of the whole arm (§8.3–8.4)
- [x] Thumb MCP universal joint (Perpendicular mate), anatomical 0–90° ranges, DOF table (§7.1)
- [ ] Re-extract after the wrist rebuild and confirm both wrist joints come out as CAD axes (§35.3)
- [ ] Enter the real wrist-flexion and thumb-lateral limits from the CAD LimitAngles
- [ ] Sweep check: every finger curls toward the palm, fingers start straight
- [ ] Palm-facing direction in MuJoCo
- [ ] 3-group controller coupling (§8.1)

## Door

- [x] Physical-door-based CAD, frame, base, door, handle
- [x] Hinge and handle references
- [x] 10 top-level door mates, detailed mate/entity macro, `LimitAngle2` / `LimitAngle3` data
- [x] Handle SolidWorks mate axis identified as `(0, 1, 0)`
- [x] Door dimensions 2.0 × 0.8 × 0.04 m, wood 600 kg/m³, mass 38.4 kg verified
- [x] Door exported through sw2robot (hinge + handle joints)
- [x] Door root rotated +90° about X so the door is vertical in MuJoCo
- [x] **v4 door**: Full_door_v4 exported, hinge + handle axes from the v4 mates (§37)
- [x] Passive door: hinge damping/friction, handle return spring preloaded against its weight
- [x] Handle ↔ frame solid collision (frame as boxes on the real frame faces), overlap diagnostics
- [x] `--check` / `--sweep` joint verification, scripted push → turn → open scene, manual keyboard control
- [ ] CAD strike hole if the handle overlaps the frame at rest (§37.5)

## Combined simulation

- [x] Combined Hand + Door MJCF prepared (with the old 15-joint hand)
- [x] Unique root-body naming
- [ ] **Regenerate the combined scene with the v4 hand (18 actuators)**
- [ ] Final hand placement relative to the handle
- [ ] Verify hand/handle contact, tune collision filters, contact/friction
- [ ] Verify door hinge behaviour under contact forces
- [ ] Door manipulation controller
- [ ] Train/validate RL interaction task

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
sw2+mujoko+python/Hand_forearm.joints.yaml          joint config (repo copy)
Mujoko/Hand_forearm_v4/mjcf/Hand_forearm.xml         sw2robot export
Mujoko/Hand_forearm_v4/mjcf/Hand_forearm_actuated.xml  current simulation model
Mujoko/Hand_forearm_v4/mjcf/hand_config.json         controller joint map
```

Tools:

```text
sw2+mujoko+python/drop_angle_mates.py
sw2+mujoko+python/fix_axes.py
sw2+mujoko+python/add_actuators.py
sw2+mujoko+python/hand_controller.py
```

Door:

```text
Door/Full_door_v3.SLDASM
Full_door_v3.joints.yaml
Full_door_v3.xml
```

Combined simulation:

```text
Mujoko/Hand_Door/Hand_Door.xml     (old hand — regenerate, §35.8)
```

Mate documentation/macros:

```text
Complete Hand + Forearm Mate Tree (52 mates)
Detailed Door Mate/Entity Report
```

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

5. The current hand reference is `Mujoko/Hand_forearm_v4/mjcf/Hand_forearm_actuated.xml` (regenerated by the §35.3 pipeline). Do not hand-edit it — change the CAD, the joints yaml or the `add_actuators.py` options and regenerate.

6. The combined `Hand_Door.xml` should remain a separate simulation-specific file so the clean hand and door exports can always be regenerated independently.

7. After every re-extract run `drop_angle_mates.py` and `fix_axes.py --reset --write` before building — axis overrides are world coordinates and go stale when the CAD moves.

8. Always pass the absolute package path to `sw2robot.exporter.build` (a relative `.` breaks mesh conversion).

# 32. Latest Session — 2026-09-26: Combined Hand + Door Scene

## 32.1 Coordinate convention

The project uses two coordinate-system conventions depending on the tool:

```text
SolidWorks:
  Vertical = Y

sw2robot / MuJoCo:
  Vertical = Z
```

Therefore, whenever the project documentation refers to **vertical** in the MuJoCo scene, it means the **MuJoCo Z axis**, even though the corresponding SolidWorks direction is Y.

The hand's current root orientation is retained from the validated hand export. The door was rotated as a complete root assembly by +90 degrees about the MuJoCo X axis so that its physical vertical direction is aligned with MuJoCo Z:

```xml
<body name="base_link" pos="0 0 0.09" euler="1.57079632679 0 0">
```

This is a scene-orientation correction; it does not change the underlying SolidWorks CAD.

## 32.2 Door physical dimensions and mass sanity check

The physical door slab dimensions provided for the simulation are:

```text
Length    = 2.00 m
Width     = 0.80 m
Thickness = 0.04 m
```

Volume:

```text
2.00 × 0.80 × 0.04 = 0.064 m³
```

The selected custom wood density is:

```text
600 kg/m³
```

Therefore the expected door-slab mass is:

```text
0.064 × 600 = 38.4 kg
```

The current MuJoCo door XML contains:

```xml
mass="38.4"
```

for `door_1`.

The very large mass previously shown for `base_1` is not treated as the door-slab mass. `base_1` is part of the fixed/base structure and should not be interpreted as the physical door itself.

## 32.3 Door SolidWorks mate extraction

The top-level door assembly is:

```text
Door/Full_door_v3.SLDASM
```

Components:

```text
base-1
frame-1
door-1
handle-1
```

The complete top-level door mate list extracted by the macro is:

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

Total:

```text
10 top-level mates
```

### Detailed `LimitAngle2`

The detailed mate/entity macro returned:

```text
ENTITY 0
Component = door-1
Reference type = face
Point = (1970.350043, 4248.670834, 3703.899707) mm
Vector = (0.997344679, 0, 0.072825756)

ENTITY 1
Component = frame-1
Reference type = face
Point = (1970.350043, 4248.670834, 3703.899707) mm
Vector = (1, 0, 0)
```

This is one of the angular constraints associated with the door/frame mechanism.

### Detailed `LimitAngle3` — handle

The detailed mate/entity macro returned:

```text
ENTITY 0
Component = door-1
Reference type = face
Point = (1172.474300, 4248.670834, 3645.639102) mm
Vector = (0.997344679, 0, 0.072825756)

ENTITY 1
Component = handle-1
Reference type = face
Point = (1322.498557, 3242.175566, 3656.593820) mm
Vector = (0, 1, 0)
```

The important handle result is:

```text
SolidWorks handle-axis vector = (0, 1, 0)
```

This is SolidWorks Y, which is the vertical convention for the CAD model and corresponds to MuJoCo Z for the project coordinate mapping.

The handle joint limits extracted into the sw2robot configuration are:

```yaml
lower: -1.57080
upper: 0.00000
```

which corresponds to approximately:

```text
-90° to 0°
```

## 32.4 Door sw2robot extraction

The door was extracted from:

```text
C:\Users\naren\Documents\Capstone\Human_Hand_200mm\Door\Full_door_v3.SLDASM
```

The extraction reported:

```text
4 components
2 limit-mate joints
4 meshes exported
5 meshes verified
```

The generated kinematic tree is:

```text
base/frame structure
    |
    +-- door_1       revolute
            |
            +-- handle_1   revolute
```

The active handle is the handle side used for the current hand-door interaction task. The opposite-side handle does not need independent interaction/control in the first simulation.

## 32.5 Door MJCF

Current door model name:

```text
Full_door_v3
```

Current dynamic joints:

```text
frame_1__door_1
    type = hinge
    range = -1.64369  1.49791 rad

 door_1__handle_1
    type = hinge
    range = -1.5708  0 rad
```

Current door actuators:

```text
frame_1__door_1_act
 door_1__handle_1_act
```

The door root includes:

```xml
euler="1.57079632679 0 0"
```

so the door is vertical in MuJoCo Z.

## 32.6 Combined Hand + Door MJCF

A separate combined scene was created:

```text
Mujoko/Hand_Door/Hand_Door.xml
```

The scene contains both articulated systems in a single MuJoCo world.

Conceptual structure:

```text
world
├── door_base
│   └── door_1
│       └── handle_1
│
└── hand_base
    └── Hand_forearm hierarchy
        ├── thumb
        ├── index
        ├── middle
        ├── ring
        └── pinky
```

Current combined model contents:

```text
Door meshes  = 8 STL files
Hand meshes  = 36 STL files
Total meshes = 44

Door joints       = 2
Hand joints       = 15
Total joints      = 17

Door actuators    = 2
Hand actuators    = 15
Total actuators   = 17
```

The hand model remains the validated `Hand_forearm` XML structure with 15 joints and 15 position actuators.

## 32.7 Mesh asset structure for combined scene

The combined MJCF uses separate asset folders so door and hand meshes do not conflict:

```text
Mujoko/
└── Hand_Door/
    ├── Hand_Door.xml
    └── assets/
        ├── door/
        │   ├── frame_1_0.stl
        │   ├── base_1_1.stl
        │   ├── frame_1_2.stl
        │   ├── base_1_3.stl
        │   ├── door_1_4.stl
        │   ├── door_1_5.stl
        │   ├── handle_1_6.stl
        │   └── handle_1_7.stl
        │
        └── hand/
            ├── fore_arm_1_0.stl
            ├── wrist_1_1.stl
            ├── human_hand_v3__Palm_1_2.stl
            ├── ...
            └── wrist_1_4.stl
```

The current working folder contains all 8 door meshes and all 36 hand meshes copied from the individual sw2robot packages.

## 32.8 Combined-MJCF duplicate-name fix

The first combined XML attempt produced a MuJoCo error because both the door and hand root bodies contained geometry names such as:

```text
base_link_visual0
base_link_visual1
base_link_collision0
base_link_collision1
```

The door root geometry was renamed to:

```text
door_base_visual0
door_base_visual1
door_base_collision0
door_base_collision1
```

The hand root geometry was then given distinct names:

```text
hand_base_visual0
hand_base_visual1
hand_base_visual2
hand_base_collision0
hand_base_collision1
hand_base_collision2
```

This avoids duplicate MuJoCo element names while leaving mesh filenames unchanged.

## 32.9 Current combined-scene placement status

The combined scene now loads both models in the same MuJoCo window.

The current issue is spatial placement: the hand and door currently intersect in the initial pose.

This is a **placement problem**, not a SolidWorks geometry problem.

The intended next change is to adjust only the `hand_base` world `pos` (and, if necessary, its final orientation) so that the hand begins near the active handle without intersecting the door.

Do not remodel the SolidWorks door or hand for this placement issue.

## 32.10 Collision intent

The physical task requires:

```text
Hand ↔ active handle       important contact
Hand ↔ door                possible contact
Handle ↔ frame             must not pass through frame
Door ↔ frame               must not block intended hinge motion
```

The door/frame CAD overlap is acceptable in the CAD representation. MuJoCo collision filtering/contact settings will be used to prevent irrelevant contacts from blocking the intended hinge movement.

## 32.11 Current hand control plan

The hand retains all 15 physical joint DOFs in the MJCF, but the high-level controller is intended to expose three control groups:

```text
Group 1 → Index
    Index MCP/PIP/DIP

Group 2 → Middle + Ring + Pinky
    corresponding MCP commands shared
    corresponding PIP commands shared
    corresponding DIP commands shared

Group 3 → Thumb
    Thumb CMC
    Thumb MCP
    Thumb IP
```

The three physical fingers in Group 2 remain separate bodies in the physics simulation.

## 32.12 Current project state at end of session

```text
[x] Human hand CAD complete for current scope
[x] Wrist/forearm assembly complete
[x] Hand mate hierarchy extracted
[x] Hand exported to MuJoCo
[x] Door CAD complete for current scope
[x] Door mate hierarchy extracted
[x] Detailed door mate/entity macro completed
[x] Door exported to MuJoCo
[x] Door set vertically in MuJoCo
[x] Door material density set to custom wood approximation
[x] Door slab mass verified at 38.4 kg
[x] Hand and door combined into one MJCF
[x] Door STL assets copied
[x] Hand STL assets copied
[x] Duplicate combined-MJCF geometry names fixed
[ ] Position hand correctly relative to active handle
[ ] Verify handle/hand contact
[ ] Tune collision groups and contacts
[ ] Tune friction/damping for door manipulation
[ ] Couple the high-level hand controls
[ ] Implement door push/pull controller
[ ] Begin RL task setup
```

# 33. Git Commands — Commit Latest Changes

Run these commands from the repository root:

```powershell
cd "C:\Users\naren\Documents\Capstone\Human_Hand_200mm"
```

Check what changed:

```powershell
git status
```

Add the updated README and combined simulation:

```powershell
git add README.md

git add Mujoko/Hand_Door/
```

Review what will be committed:

```powershell
git status
```

Commit:

```powershell
git commit -m "Add combined hand and door MuJoCo scene"
```

Push to the main branch:

```powershell
git push origin main
```

Verify the commit after pushing:

```powershell
git log -1 --oneline
```

## If the asset folder is ignored or not staged

Check:

```powershell
git status --ignored
```

If the STL files are intentionally part of the repository and are not ignored, they can be added with:

```powershell
git add Mujoko/Hand_Door/assets/
git commit -m "Add Hand Door MuJoCo mesh assets"
git push origin main
```

Do not force-add files merely to bypass `.gitignore` without first checking whether the repository is intended to track generated STL assets.

## Commit the RL environment

```powershell
cd "C:\Users\naren\Documents\Capstone\Human_Hand_200mm"
git add README.md "sw2+mujoko+python/door_env.py" "sw2+mujoko+python/train_policy.py" "sw2+mujoko+python/door_setup.py" "sw2+mujoko+python/door_scene.py"
git add "Mujoko/Full_door_v4/"
git status          # sw2+mujoko+python/output/, runs/ and policies/ should not be listed
git commit -m "Gymnasium env: hand + door scene, tasks reach/handle/door, coupled middle-ring-pinky; handle stays upright"
git push origin main
```

## Commit the door work (2026-09-28)

```powershell
cd "C:\Users\naren\Documents\Capstone\Human_Hand_200mm"
git status
git add README.md
git add "sw2+mujoko+python/door_setup.py" "sw2+mujoko+python/door_scene.py" "sw2+mujoko+python/fix_axes.py"
git add "sw2+mujoko+python/Full_door_v4.joints.yaml"
git add "Mujoko/Full_door_v4/"
git add Door/
git status
git commit -m "Door v4 in MuJoCo: upright, masses, passive hinge + handle spring, solid handle-frame collision, scene + manual control"
git push origin main
```

## Commit this session (2026-09-27)

```powershell
cd "C:\Users\naren\Documents\Capstone\Human_Hand_200mm"
git status
git add README.md "sw2+mujoko+python/" "Mujoko/Hand_forearm_v4/"
git add Hand_forearm.SLDASM "human hand v3.SLDASM" 01_Palm/ 06_Thumb/
git status
git commit -m "Full 18-actuator hand export: ball-centre wrist, sw2robot fixes, actuator + controller tools"
git push origin main
```

Add the wrist and forearm part files too if they live outside the folders above (`git status` lists them). Check `git status --ignored` before force-adding generated STL files.

# 34. Reproducible Combined-Scene Launch

From the combined scene directory:

```powershell
cd "C:\Users\naren\Documents\Capstone\Human_Hand_200mm\Mujoko\Hand_Door"
```

Launch:

```powershell
python -m mujoco.viewer --mjcf ".\Hand_Door.xml"
```

Required folder layout:

```text
Hand_Door/
├── Hand_Door.xml
└── assets/
    ├── door/
    └── hand/
```

If MuJoCo reports a missing mesh, first verify the corresponding STL exists in the correct `assets` subfolder before changing the MJCF mesh paths.

# 35. Session — 2026-09-27: Full Hand Export (sw2robot → MuJoCo, 18 actuators)

## 35.1 Goal

Export the complete SolidWorks hand (every finger joint, thumb, wrist) to MuJoCo with sw2robot, add actuators, and make it fully controllable with a gamepad.

## 35.2 Problems found and how they were fixed

sw2robot version: **0.4.4** (`python -m pip install sw2robot`, Python 3.12). The editor app and the `python -m sw2robot…` commands are separate installs.

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | Editor showed **10 links / 6 joints**; each finger one rigid block | `expand:` entries used underscores (`index_assem_2`), but sw2robot matches them as substrings of the SolidWorks name (`human hand v3-2/index_assem-2`, hyphens) → no match. The thumb only expanded because its parts were under-defined | `expand:` now lists `index_assem`, `Middle_Finger`, `Ring_finger`, `Pinky_finger`, `Thumb_finger` (no suffix) |
| 2 | Every joint "**no CAD axis, defaulting to world +Z**" | Inside sub-assemblies sw2robot reads a **LimitAngle as a rigid, fixed angle** → removes the hinge's one free rotation. Top-level limit mates are read properly, but they referenced `human hand v3-2`, which is expanded away | `drop_angle_mates.py` deletes the ANGLE mate records from `graph.json`; limits live in the joints yaml instead |
| 3 | Ring and pinky joints still had no axis | Their mates reach sw2robot as point + reference-axis geometry its classifier reports as "under-constrained" | `fix_axes.py` reads the mated reference axis and writes `axis_point` / `axis_dir` into the yaml |
| 4 | Thumb Base→Proximal had no axis | Mated point-only (it carries 2 LimitAngles for flexion + lateral) | `fix_axes.py`: flexion axis parallel to the thumb IP axis; lateral axis added in MuJoCo (#8) |
| 5 | MJCF build crashed: `could not find ../meshes/fore_arm_1.3dxml` | Passing `.` as the package path breaks sw2robot's "is the mesh inside the package" check, so meshes were never converted | Always pass the **absolute** package path (`$pkg`) |
| 6 | Hand lay horizontal (+Y) in MuJoCo | SolidWorks vertical is Y (§3) | `add_actuators.py --point-up` rotates the root so forearm → hand is +Z and rests the forearm on the floor |
| 7 | Palm "dislocated" from the wrist ball when flexing | Pivot was the mate point on the ball **surface** | Wrist rebuilt in CAD with both axes through the ball centre (§6) |
| 8 | Palm tilt and thumb lateral motion missing | URDF allows **one axis per joint**; the second LimitAngle on a joint cannot become a joint | Wrist: second hinge made in CAD (forearm↔ball). Thumb: `--add-axis "Thumb_Proximal_1@LO,HI"` adds a second hinge at the same pivot in MuJoCo |
| 9 | First guessed wrist axis duplicated the forearm→wrist axis | Heuristic "parallel to next joint" picked the thumb CMC axis | Replaced by the CAD ball-centre wrist; `--reset` drops old overrides |
| 10 | "Available committed memory is critically low" with 16 GB RAM free | Commit limit = RAM + page file; page file was only 2 GB and sw2robot keeps a **hidden second SolidWorks** running | Page file set to 16 384 – 32 768 MB on C:; close the extra `SLDWORKS.exe` / the sw2robot editor when not extracting |
| 11 | Thumb Base→Proximal looked **twisted** | Point-only mate (ball joint) → bend axis was only estimated (∥ IP), so straightening/bending rotated about a slightly wrong axis | **Perpendicular mate** in `Thumb_finger.SLDASM` between `thumb_proxiam_pip+right` (base) and `Thumb Proximal_pip+front` (proximal) = universal joint; `fix_axes.py` reads the bend axis from it; `add_actuators.py --square-axis Thumb_Proximal_1` makes that axis exactly ⟂ to the proximal bone |
| 12 | CAD changes had no effect | The sw2robot editor started from `sw2+mujoko+python` writes to **`sw2+mujoko+python\output\`**, while the commands still read the old Temp package; also only **Save All** saves sub-assembly mates | Always use the `output\` paths of §35.3; check `graph.json`'s time after extracting |
| 13 | Fingers started at 45° and bent backwards | Joint 0 = extraction pose; hinge axes point either way | `add_actuators.py` measures each joint's CAD bend, flips axes so + = flexion and sets `ref` → **0° = straight, 90° = bent** |
| 14 | Thumb bent backwards (changes with dimensions) | Thumb joints are straight in CAD, so their direction is guessed from the fingers | `--flip NAME` per thumb joint (check `--pose 45` after every rebuild) |
| 15 | Whole arm tilted | `--point-up` aimed forearm→hand centre of mass | Now uses the forearm cylinder's own long axis |

## 35.3 Pipeline (run every time the CAD changes)

```powershell
# 0. SolidWorks: File -> Save All (Save All, so sub-assembly mates are saved too), then close SolidWorks
# 1. start the editor FROM the scripts folder (so it writes to sw2+mujoko+python\output):
cd "C:\Users\naren\Documents\Capstone\Human_Hand_200mm\sw2+mujoko+python"
sw2robot-web.exe      # browser: open Hand_forearm.SLDASM -> "drop meshes & re-extract"; then Ctrl+C here

# 2. new PowerShell window
cd "C:\Users\naren\Documents\Capstone\Human_Hand_200mm\sw2+mujoko+python"
$pkg = "C:\Users\naren\Documents\Capstone\Human_Hand_200mm\sw2+mujoko+python\output\Hand_forearm"
$mj  = "C:\Users\naren\Documents\Capstone\Human_Hand_200mm\sw2+mujoko+python\output\Hand_forearm_mjcf"
$dst = "C:\Users\naren\Documents\Capstone\Human_Hand_200mm\Mujoko\Hand_forearm_v5"     # new folder per model version
$yml = "$pkg\Hand_forearm.joints.yaml"
(Get-Item "$pkg\graph.json").LastWriteTime                  # must be the extraction you just did

# 3. make the axes resolvable
Copy-Item ".\Hand_forearm.joints.yaml" $yml -Force          # the repo copy has the expand list + limits
python drop_angle_mates.py $pkg
python fix_axes.py $pkg --reset --write                      # thumb MCP: "Perpendicular mate ... = BEND axis"

# 4. build URDF + MJCF (absolute path!)
python -m sw2robot.exporter.build $pkg --config $yml --mujoco --mujoco-fixed-base

# 5. copy the model into the repo
New-Item -ItemType Directory $dst -Force | Out-Null
Copy-Item "$mj\*" $dst -Recurse -Force
Copy-Item $yml ".\Hand_forearm.joints.yaml" -Force

# 6. actuators (final options), then check
python add_actuators.py "$dst\mjcf\Hand_forearm.xml" --point-up --fix-wrist --square-axis Thumb_Proximal_1 --add-axis "Thumb_Proximal_1@-45,45"
python hand_controller.py "$dst\mjcf\Hand_forearm_actuated.xml" --pose 45     # every joint toward the palm? thumb not twisted?
python hand_controller.py "$dst\mjcf\Hand_forearm_actuated.xml" --sweep
```

If a thumb joint bends backwards at `--pose 45`, add `--flip Thumb_Base_1`, `--flip Thumb_Proximal_1` or `--flip Thumb_Distal_v2_1` to step 6 and rerun it (this can change when dimensions change). Copy `poses.json` from the previous model folder to keep saved poses. `sw2+mujoko+python/output/` is generated — it is in `.gitignore`.

`--reset` matters: `axis_point` / `axis_dir` overrides are **world coordinates**. If the CAD moved (e.g. the palm straightened by ~8° after the wrist rebuild), old overrides point at the wrong place.

Expected results:

```text
build          : expanding sub-assembly … ×6, "17 revolute", "18 links, 17 joints", no "no CAD axis" notes
fix_axes       : wrist joints through the ball centre (wrist points near Y 1032, not 1062);
                 Thumb_Base_1 -> Thumb_Proximal_1: "Perpendicular mate (universal joint) ... = BEND axis",
                 bend axis ~0 deg to the IP hinge, sideways axis ~90 deg
add_actuators  : [square-axis] …, [add-axis] …Thumb_Proximal…_abd, [anatomical] 0 = straight lines,
                 [wrist] pivots 0.0 mm from the ball centre, [point-up] forearm long axis now [0, 0, 1],
                 wrist {'flex': …, 'deviation': …}, spread [… _abd], "18 actuators"
```

## 35.4 Joint configuration (`Hand_forearm.joints.yaml`)

```yaml
base: fore_arm_1
expand: [index_assem, Middle_Finger, Ring_finger, Pinky_finger, Thumb_finger]
joints:                                   # parent -> child, type, limits (rad) used by sw2robot
  fore_arm_1 -> wrist_1                       revolute  -0.17453 ... 0.17453   wrist tilt +-10 deg
  wrist_1 -> human_hand_v3_2__Palm_1          revolute  -1.0 ... 1.0            wrist flexion (set from LimitAngle)
  Palm -> Thumb_Base, Thumb_Base -> Thumb_Proximal, Thumb_Proximal -> Thumb_Distal_v2    revolute
  Palm -> <finger>_Proximal -> <finger>_Middle -> <finger>_Distal                        revolute (x4 fingers)
```

Full link names are `human_hand_v3_2__<sub-assembly>_<n>__<part>_<n>`, e.g. `human_hand_v3_2__index_assem_2__Index_Proximal_1`. `fix_axes.py --write` adds `axis_point` / `axis_dir` to the ring, pinky and thumb-MCP entries.

The wrist limits in the yaml are used as they are. **For the fingers and thumb the MuJoCo ranges are set by `add_actuators.py`** (0 = straight, 90° = bent — `--flex-range`, `--cmc-range`), so the finger limits in the yaml only matter for the URDF / sw2robot editor.

## 35.5 Actuator list (18)

Actuator name = `act_<MJCF joint name>`; the added lateral hinge is `<thumb MCP joint>_abd`. sw2robot names joints after the links they connect (the door export shows the `parent__child` form, e.g. `frame_1__door_1`); the exact names are printed by `add_actuators.py` and stored in `hand_config.json`.

| # | Group | Joint (parent → child link) | Motion | Axis source | Range |
|---:|---|---|---|---|---|
| 1 | wrist | fore_arm_1 → wrist_1 | tilt (left/right) | CAD, ball centre | ±10° |
| 2 | wrist | wrist_1 → Palm_1 | flexion (front/back) | CAD, ball centre | from LimitAngle (yaml placeholder ±57°) |
| 3 | thumb | Palm_1 → Thumb_Base_1 | CMC | CAD | 0° = in palm plane … 90° |
| 4 | thumb | Thumb_Base_1 → Thumb_Proximal_1 | MCP bend | Perpendicular mate (base axis), squared to the bone | 0 … 90° |
| 5 | thumb | Thumb_Proximal_1 (added `_abd`) | **MCP sideways** | `--add-axis`, same pivot, ⟂ bend axis | −45 … +45° (LimitAngle3) |
| 6 | thumb | Thumb_Proximal_1 → Thumb_Distal_v2_1 | IP | CAD | 0 … 90° |
| 7–9 | index | Palm → Proximal → Middle → Distal | MCP, PIP, DIP | CAD | 0 … 90° each (0 = straight) |
| 10–12 | middle | Palm → Proximal_5 → Middle_4 → Distal_4 | MCP, PIP, DIP | CAD | same |
| 13–15 | ring | Palm → Proximal → Middle → Distal | MCP, PIP, DIP | fix_axes (mated ref. axis) | same |
| 16–18 | pinky | Palm → Proximal → Middle → Distal | MCP, PIP, DIP | fix_axes (mated ref. axis) | same |

Gains: §23. Each actuator also has a `jointpos` sensor `q_<joint>`.

## 35.6 Controller

See §8.2 for the gamepad mapping. `--sweep` moves one joint at a time and prints its name (use it after every rebuild to check directions), `--demo` cycles open/fist/pinch/point, `--max-rate DEG_PER_S` slows every commanded motion.

## 35.7 Verification checklist after each rebuild

1. Build output has no "no CAD axis" notes.
2. `add_actuators.py` reports 18 actuators and the expected wrist/thumb roles.
3. Sweep: every finger curls toward the palm; fingers start straight (else give that joint a negative `lower`).
4. Sweep: the palm rolls around the wrist ball without a gap in both flexion and tilt.
5. Sweep: the thumb lateral hinge swings the thumb toward/away from the index finger (if it moves in the wrong plane, give an explicit axis: `--add-axis "Thumb_Proximal_1:x,y,z@LO,HI"`, body frame).

## 35.8 Next steps / ideas

1. Enter the real wrist-flexion limits (yaml) from its LimitAngle (currently ±57°).
2. Make the thumb direction automatic (always toward the palm) so `--flip` is never needed after dimension changes.
3. Map the 3 control groups of §8.1 onto the gamepad / keyboard (index separate from middle+ring+pinky).
4. Regenerate `Hand_Door.xml` with the v5 hand (18 + 2 = 20 actuators), `hand_base` / `door_base` naming (§32.8); use keyboard mode to bring the hand to the handle.
5. Fingertip contact: friction / softer contact (`condim 4`, `friction`) on the distal pads for a stable grip on the handle.
6. Pose sequences: approach → pre-shape → close → turn, played from saved poses (basis for scripted demos and RL resets).
7. Record & replay (log `qpos`/`ctrl` to CSV) for analysis and imitation data.

# 36. Tool Reference (`sw2+mujoko+python/`)

Requirements: `python -m pip install sw2robot mujoco numpy pygame pyyaml`

## 36.1 `drop_angle_mates.py`

```text
python drop_angle_mates.py <pkg_dir>            remove ANGLE (LimitAngle) records from graph.json
python drop_angle_mates.py <pkg_dir> --restore  put the extracted graph.json back
```

Backs up each fresh extraction to `graph.json.bak`. Run once after every re-extract.

## 36.2 `fix_axes.py`

```text
python fix_axes.py <pkg_dir>                    report: OK / FIX per joint, the mate geometry, a proposed axis
python fix_axes.py <pkg_dir> --write            write axis_point / axis_dir for FIX joints into the yaml (.bak kept)
python fix_axes.py <pkg_dir> --reset --write    drop all old overrides first (use after a re-extract)
```

Universal joints (point Coincident + Perpendicular between two axes, no axis-to-axis Coincident) always take their bend axis from the mate — the axis closest to the next joint's hinge (`--uj-bend parent|child` to force one side). `--ref-axis CHILD=AXIS` takes an axis from a reference axis drawn in the top assembly; `--list-axes` lists them. Other proposal order: mated reference axis → axis parallel to the neighbouring joint (if several axes are mated) → parallel to the next joint in the chain → perpendicular to the previous joint. Uses sw2robot's own graph loader, so it sees exactly what the build sees. The rewritten yaml loses comments.

## 36.3 `add_actuators.py`

```text
python add_actuators.py <mjcf.xml> [options]    → <name>_actuated.xml + hand_config.json (same folder)
  --point-up                     forearm → palm along +Z, forearm resting on the floor
  --add-axis NAME[:x,y,z][@lo,hi]  second hinge in the body of NAME (part of a joint/body name),
                                 auto axis ⟂ existing axis and bone; range in degrees
  --fix-wrist                    rebuild both wrist axes from the hand (flexion ∥ knuckles, tilt ⟂ palm)
  --square-axis NAME             make a hinge axis exactly ⟂ to its bone (thumb MCP: removes the twist)
  --flip NAME                    reverse a joint's bending direction
  anatomical zero (default)      0 = straight, + = flexion; --flex-range LO HI (default 0 90), --cmc-range LO HI
  --straight-tol DEG             joints bent less than this in CAD keep the CAD pose as 0 (default 10)
  --keep-cad-zero NAME / --set-zero NAME=DEG / --start-cad NAME   manual control of a joint's 0 / start pose
  --no-anatomical                keep the exported zero and axis signs
  --pivot-to-parent-center BODY  move BODY's joints to its parent's centre of mass (not needed with the CAD ball wrist)
  --weld-base                    remove a free joint if exported without --mujoco-fixed-base
  --kp / --sag / --kp-min / --zeta / --force-margin   gain tuning (§23)
  --keep-contacts                don't exclude pairs touching in the open pose
```

Groups joints automatically: palm = body the fingers hang off; each chain named by the earliest finger keyword in its link names; joints above the palm = wrist (2 CAD wrist hinges → `flex` + `deviation`).

## 36.4 `hand_controller.py`

```text
python hand_controller.py <actuated.xml>             gamepad control (§8.2)
python hand_controller.py <actuated.xml> --sweep     one joint at a time
python hand_controller.py <actuated.xml> --demo      preset cycle
python hand_controller.py <actuated.xml> --max-rate 90   slower motions (deg/s, default 344)
python hand_controller.py <actuated.xml> --demo --headless 10   no window; prints tracking error
python hand_controller.py <actuated.xml> --pose 45 [--only NAME]  hold all (or only NAME) joints at 45 deg
python hand_controller.py <actuated.xml> --keyboard   move the whole arm + pose keys (§8.3)
python hand_controller.py <actuated.xml> --save-pose NAME / --load-pose NAME / --list-poses   (§8.4)
```

Reads `hand_config.json` from the model folder; edit `open` / `closed` / `sign` per joint there to change bend direction or lateral sign without rebuilding.

## 36.5 `check_pose.py`

```text
python check_pose.py <mjcf folder> [--pair PARENT CHILD]
```

Compares each part's rotation relative to its parent in the raw sw2robot export (CAD pose) and in the actuated model's start pose; **TWIST** = any rotation not about the joint's own hinge axes (should be 0).

## 36.6 `door_setup.py`

```text
python door_setup.py <Full_door_v4.xml> [options]      -> Full_door_v4_sim.xml + door_config.json (same folder)
  --mass PART=KG               set a part's mass, inertia scaled with it (door=38.4, handle=0.5)
  --handle-hits-frame          frame collision rebuilt as boxes matching the real frame (jambs, header),
                               handle <-> frame collision always ON, constraint latch removed
  --pocket                     with --handle-hits-frame: cut a pocket where the CAD handle passes through the frame
  --up y|hinge|plate           which direction becomes vertical (default y = SolidWorks up)
  --door-damping / --door-friction / --handle-spring / --handle-damping   door feel
  --unlock-deg / --no-latch / --keep-latch                                constraint latch options
  --handle-stays               no return spring: the handle stays where it is turned (weightless + light friction)
  --flip-handle / --handle-range LO HI                                    handle direction / range
python door_setup.py <Full_door_v4_sim.xml> --check    measure hinge + handle: axis, pivot, range, collisions
python door_setup.py <Full_door_v4_sim.xml> --sweep    animate both joints through their ranges
python door_setup.py <Full_door_v4_sim.xml> --view-only
```

## 36.7 `door_scene.py`

```text
python door_scene.py <Full_door_v4_sim.xml>            scripted: PUSH (handle hits frame) -> TURN (lever vertical) -> OPEN
python door_scene.py <sim.xml> --once                  run it once
python door_scene.py <sim.xml> --manual                drive the door and handle yourself (§37.6)
  --open-dir 1|-1   --push N*m   --door-speed rad/s   --turn-speed deg/s   --push-time s   --pause s
```

## 36.8 PowerShell notes

- `<...>` in instructions is a placeholder — PowerShell treats `<` as an operator. Use the `$pkg` / `$dst` variables.
- Run the scripts from `sw2+mujoko+python`, or give their full path.
- An option like `--only "…"` only works at the end of a command, never on its own line.
- Downloaded script files land in `Downloads` — copy them into `sw2+mujoko+python` (check with `python <script> -h`).
- Start `sw2robot-web.exe` from `sw2+mujoko+python`, so packages go to `sw2+mujoko+python\output\`. Stop it with Ctrl+C when the extraction is done (it keeps a hidden SolidWorks running).
- A `Traceback … KeyboardInterrupt / time.sleep` after Ctrl+C or closing a viewer is normal.


# 37. Session — 2026-09-28: Door (`Full_door_v4`) in MuJoCo

## 37.1 CAD

Assembly `Door/Full_door_v4.SLDASM`, parts `base`, `frame`, `door`, `handle`. Mates (10):

| Pair | Mates | Joint |
|---|---|---|
| base ↔ frame | Coincident1 (plane), Coincident2 (point), Coincident3 (axis) | fixed |
| frame ↔ door | Coincident5 (hinge point), Coincident7 (hinge axis, SolidWorks Y = vertical), LimitAngle1 | **hinge** (vertical) |
| door ↔ handle | Coincident8 (handle centre), Coincident9 (axis through the door), LimitAngle2 | **handle** (turns in the door plane) |

The mate report we first used was titled **Full_door_v3**; in v4 the door is turned 90° relative to v3, so v3's axis numbers must not be used for v4 (§37.3, problem 5). Parts have no material in CAD → masses are set by `door_setup.py`.

## 37.2 Pipeline

```powershell
# 1. SolidWorks: open Door\Full_door_v4.SLDASM -> File -> Save All -> close SolidWorks
# 2. extract (browser: open ...\Door\Full_door_v4.SLDASM -> extract; then Ctrl+C here)
cd "C:\Users\naren\Documents\Capstone\Human_Hand_200mm\sw2+mujoko+python"
sw2robot-web.exe

# 3. build + set up (new PowerShell window)
cd "C:\Users\naren\Documents\Capstone\Human_Hand_200mm\sw2+mujoko+python"
$pkg = "C:\Users\naren\Documents\Capstone\Human_Hand_200mm\sw2+mujoko+python\output\Full_door_v4"
$mj  = "C:\Users\naren\Documents\Capstone\Human_Hand_200mm\sw2+mujoko+python\output\Full_door_v4_mjcf"
$dst = "C:\Users\naren\Documents\Capstone\Human_Hand_200mm\Mujoko\Full_door_v4"
$yml = "$pkg\Full_door_v4.joints.yaml"

Copy-Item ".\Full_door_v4.joints.yaml" $yml -Force
python drop_angle_mates.py $pkg
python fix_axes.py $pkg --reset --from-mates door_1 --from-mates handle_1 --write
python -m sw2robot.exporter.build $pkg --config $yml --mujoco --mujoco-fixed-base
New-Item -ItemType Directory $dst -Force | Out-Null
Copy-Item "$mj\*" $dst -Recurse -Force

python door_setup.py "$dst\mjcf\Full_door_v4.xml" --mass door=38.4 --mass handle=0.5 --handle-hits-frame
python door_setup.py "$dst\mjcf\Full_door_v4_sim.xml" --check
python door_scene.py "$dst\mjcf\Full_door_v4_sim.xml" --manual
```

`Full_door_v4.joints.yaml`: `base: base_1`; `base_1 → frame_1` fixed; `frame_1 → door_1` revolute; `door_1 → handle_1` revolute. The axes/pivots are written into it by `fix_axes.py --from-mates` from the **v4** mates (not typed in by hand).

## 37.3 Problems found and fixes

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | Viewer showed a white/black split screen | Camera inside the (large, light) base plate | **Align** in the viewer; correct upright (#2) |
| 2 | Base and door parallel, door lying flat | Model was stood up by the hinge axis, which came out wrong (LimitAngle), and later by the "flattest" base direction, which the merged frame spoils | `door_setup.py` stands the model up by **SolidWorks Y** (`--up y`, default) and warns if the hinge is not vertical |
| 3 | Hinge axis wrong in the export | LimitAngle mates read as rigid (same as the hand) | `drop_angle_mates.py` + `fix_axes.py --from-mates` |
| 4 | Handle axis `[0,0,1]` vs mate `(-1,0,0)` | LimitAngle2 compares two **opposite** lines → sw2robot cannot derive an axis from it | axis taken from the handle's own coincident axis mate (`--from-mates handle_1`) |
| 5 | Handle turned in the wrong plane (swung out of the door) | Axes typed in from the **v3** mate report; v4's door is turned 90° | removed; `--from-mates` reads the v4 mates |
| 6 | Door jammed at the hinge while swinging | MuJoCo does not filter parent/child contacts when the parent (frame) is fixed to the world | frame ↔ door contact always excluded |
| 7 | Handle sagged ~8° under its own weight | return spring too weak for the handle's weight | spring **preloaded** so the handle rests level (`[handle] … sag … spring preloaded`) |
| 8 | Handle "collided" with the frame everywhere / passed through it | MuJoCo collides meshes as their **convex hull**; the U-shaped frame's hull fills the door opening | frame collision rebuilt as **boxes matching the real frame** (`--handle-hits-frame`) |
| 9 | Door flung open at the start ("moved 90 deg by itself") | Handle's convex hull overlapping the frame boxes at rest → shoved apart | boxes fitted to the real frame faces; overlap measured and reported; pocket (optional) built from the handle's **hull** |
| 10 | Handle "partially merged" with the frame | Pocket = invisible hole in the frame; the visible frame has no hole | **solid frame by default**; pocket only with `--pocket` |
| 11 | Door not blocked / handle stuck | In the CAD the handle's end sits inside the frame (like a latch bolt without a strike hole) | CAD fix §37.5, or `--pocket` |

## 37.4 Door model settings (`door_setup.py`)

| Item | Value |
|---|---|
| Door mass | 38.4 kg (2.0 × 0.8 × 0.04 m wood, 600 kg/m³) — `--mass door=38.4` |
| Handle mass | 0.5 kg — `--mass handle=0.5` |
| Hinge | passive, damping 2 N·m·s/rad, friction 0.5 N·m, armature 0.01; range from LimitAngle1 |
| Handle | return spring 4 N·m/rad (preloaded against its weight), damping 0.02; range from LimitAngle2 (0 → 90°) |
| Collision | handle ↔ frame **solid** (frame = boxes on the real frame faces); door ↔ frame off; door ↔ handle off (parent/child) |
| Latch | physical (handle hits frame) with `--handle-hits-frame`; constraint latch (door locked until the handle turns 30°) otherwise |
| Sensors | `door_angle`, `handle_angle` (jointpos) |
| Orientation | SolidWorks Y → world Z, base resting on z = 0 |

`door_config.json` stores the hinge / handle joint names and latch settings for the scene and later scenes.

## 37.5 Handle ↔ frame: making both solid

`door_setup.py` measures whether the **CAD** handle intersects the real frame:

```text
[frame-boxes] ... SOLID frame, exactly the real shape (no pockets); the handle is clear of the frame at every handle angle
   -> good: the handle stops against the real frame face, nothing passes through
[frame-boxes] ... WARNING: the CAD handle OVERLAPS the frame with the door closed (at rest);
                  turning the handle with the door closed hits the frame between A and B deg
   -> two solid parts cannot start inside each other: fix in CAD
```

CAD fix (a strike hole, exactly like a real latch): with the door closed and the handle at rest, edit **frame** in context → Insert → Features → **Cavity** → design component = **handle**. Repeat at a few handle angles (e.g. 30°, 60°, 90°) to clear its turning path. Save All, re-extract, rerun §37.2. Door ↔ handle overlap is fine (they never collide).

## 37.6 Manual control (`door_scene.py --manual`)

A small **"Door manual control"** window opens next to the viewer — click it, then hold:

| Key | Action |
|---|---|
| W / ↑ | push the door open (while held) |
| S / ↓ | push it closed |
| D / → | turn the handle + (stays where you leave it) |
| A / ← | turn the handle − |
| Space | let go of the handle (its spring returns it) |
| R | reset |
| Esc | quit |

The window shows the door and handle angles and **whether the handle is touching the frame** (red when it is); the terminal prints each time it starts / stops touching. The viewer shows contact points and force arrows where the handle meets the frame.

## 37.7 Scripted scene (`door_scene.py`)

`PUSH` (2 s push → the handle hits the frame, door stops) → `TURN` (handle to vertical; the angle is computed from the handle geometry) → `OPEN` (door swings to 90 % of its limit) → pause → repeat. Before pushing it checks that nothing moves by itself (an overlap) and prints what overlaps. Options §36.7.

## 37.8 Verification

1. `--check`: hinge `0.0 deg from vertical`, door centre ≈ 400 mm from the hinge line; handle `90 deg from vertical`, `≈0 deg from the door's face normal`, pivot ≈ 1005 mm high and ≈ 650 mm from the hinge line.
2. `door_setup`: `[handle-frame] handle <-> frame collision ON, no overlap with the door closed`; no `WARNING`.
3. `--manual`: W alone → the door stops with "handle touching the frame"; D until vertical → W opens the door.

## 37.9 Next steps

1. CAD strike hole (§37.5) if the setup reports an overlap.
2. Combined scene: v5 hand (keyboard mode, soft-welded arm) in front of the door; hand grips and turns the handle, pushes the door.
3. Fingertip friction for gripping the handle; scripted reach → grip → turn → push from saved poses.


# 38. RL environment — `door_env.py` (Gymnasium, MuJoCo)

## 38.1 What it is

One MuJoCo scene built at start-up from the two models you already have:

```text
hand : Mujoko/Hand_forearm_v5/mjcf/Hand_forearm_actuated.xml + hand_config.json   (add_actuators.py)
door : Mujoko/Full_door_v4/mjcf/Full_door_v4_sim.xml        + door_config.json    (door_setup.py)
```

All names get a prefix (`hand/…`, `door/…`), mesh paths are made absolute, the source files are not changed; the merged model is also written to `Mujoko/Hand_Door_env/scene.xml`. The arm floats on a **soft weld to a mocap target** (as in keyboard mode), so it collides and pushes physically. It is placed automatically: palm facing the door, in front of the handle.

## 38.2 Tasks and actions (all actions in [-1, 1], 25 Hz control, sim 500 Hz)

| Task | Goal | Actions | Size |
|---|---|---|---|
| `reach` | palm to the handle (< 4 cm) | arm_pos (3), arm_rot (3) | 6 |
| `handle` | grasp and turn the handle ≥ 80 % | arm_pos (3), arm_rot (3), wrist (2), index, mrp, thumb, opp | 12 |
| `door` | handle already turned: door ≥ 60° | arm_pos (3), grip | 4 |

* `arm_pos`: target moves ≤ 1 cm / step; `arm_rot`: ≤ 3° / step (world axes); 5 cm leash to the real arm
* `wrist`: flexion, tilt across their ranges; `index`, `thumb`: curl 0 → 1; `opp`: thumb CMC
* **`mrp` = middle + ring + pinky coupled** (one action → all 9 joints, via the curl mapping of `hand_config.json`); `grip` = index + mrp + thumb
* Coupling is done in the action mapping (every joint keeps its own actuator); a tendon-driven version can be added if the real hand uses one motor for these fingers.

**Env latch:** the door is held closed (both directions) until the handle reaches 80 % of its travel, so the agent cannot open the door without turning the handle (the physical handle ↔ frame stop only blocks one direction).

## 38.3 Observations (flat vector)

palm → handle vector (3), palm x / z axes (6), hand joint angles normalised (18), door & handle angle + velocity (4), fingertip-on-handle contacts (5), palm-on-handle contact (1), arm target offset (3), previous action.

## 38.4 Rewards

* reach: −distance + 5 × distance improvement; +10 on success
* handle: 10 × handle-turn progress − 0.5 × distance + 0.05 per fingertip / palm contact; +10 at ≥ 80 %
* door: 5 × door-angle progress + 0.02 while touching the handle; +10 at 60°
* all: −0.01 × |action|²

## 38.5 Door behaviour used by the env

Build the door with **`--handle-stays`** so the handle stays where it is turned (upright stays upright):

```powershell
python door_setup.py "$dst\mjcf\Full_door_v4.xml" --mass door=38.4 --mass handle=0.5 --handle-hits-frame --flip-handle --handle-stays
```

It removes the return spring, makes the handle weightless (`gravcomp`) with 0.1 N·m friction: it holds exactly at 0, 20, 45, 70, 90° and a gentle 0.5 N·m turns it.

## 38.6 Commands

```powershell
cd "C:\Users\naren\Documents\Capstone\Human_Hand_200mm\sw2+mujoko+python"
python -m pip install gymnasium stable-baselines3 tensorboard

python door_env.py --task reach --check          # gymnasium env checker
python door_env.py --task reach --scripted       # hand-written baseline in the viewer
python door_env.py --task handle --scripted
python door_env.py --task door --scripted
python door_env.py --task reach                  # random actions

python train_policy.py --task reach --steps 200000
python train_policy.py --task reach --play
tensorboard --logdir runs
```

Stand-in test results: all three tasks pass the checker; scripted baselines solve reach (~28 steps), handle (~17 steps, 88–97 %, door stays shut), door (~77 steps, 60°); random actions do not.

## 38.7 Collisions, arm force limit, start pose

| Pair | Collides |
|---|---|
| hand ↔ hand | no (parts may overlap) — `--self-collision` to turn on |
| hand ↔ door / handle / frame / floor | yes — stiff contacts (overlap < 1 mm when pushing) |
| handle ↔ frame | yes (solid, `door_setup.py --handle-hits-frame`) |

The arm is gravity-compensated and follows its target through a soft weld; the target may run at most **2 cm** ahead of the arm, which caps the push at ≈ 20 N (with a 5 cm lead the pull reached ≈ 300 N and the hand tunnelled through the 4 cm door). At every reset the start pose is checked: if any hand part overlaps the door, the arm is moved back along the door normal until it is clear (`info["start_pushed_back"]`). If the hand was exported without collision shapes, its visible meshes are used and a message is printed.

Finger, thumb and wrist actions are **rates**: 0 = hold, ±1 = 10 % of the range per step (an action of 0 used to mean "half closed").

## 38.8 Domain randomization (fixed vs randomized training)

```python
DoorEnv(task="door")                                    # fixed (nominal) door
DoorEnv(task="door", randomize_physics=True)            # randomized every reset
DoorEnv(task="door", physics={"door_mass": 1.8, "hinge_friction": 3.0})   # a held-out test door
```

| Parameter (multiplier of the nominal door) | Training range |
|---|---|
| door_mass (and inertia) | 0.6 – 1.4 |
| hinge_damping | 0.5 – 2.0 |
| hinge_friction | 0.5 – 2.0 |
| handle_friction | 0.5 – 2.0 |
| grip_friction (handle surface) | 0.6 – 1.4 |

`info["physics"]` reports the values used. Test doors outside these ranges (e.g. mass 1.8) measure generalization.

## 38.9 Tests (`env_tests.py`)

```powershell
python env_tests.py                       # everything, PASS/FAIL with numbers, writes env_test_report.json
python env_tests.py --only collisions     # one group: info start collisions latch actions obs reward random speed
python door_env.py --task reach --view    # look at the start pose, no policy
```

Stand-in results: 47/47 pass — start pose clear of the door (20 starts per task), 0 hand self-contacts in a fist, hand stopped by the door (0.8 mm overlap), door shut while the handle is not turned (0.14°), door opens once it is (60°), every action channel moves only its own joints, observations finite and bounded, reward prefers moving toward the handle, randomization within its ranges, ≈ 1000 env steps/s (1 M steps ≈ 0.3 h).

## 38.10 Next steps

1. Train `reach` → `handle` → `door`; start `handle` / `door` episodes from the end states of the previous policy.
2. A switcher: palm within 4 cm → handle policy; handle ≥ 80 % → door policy.
3. If the real hand uses one cable motor for middle+ring+pinky: switch `mrp` to a MuJoCo tendon (adaptive grip).
4. Randomise door mass, handle friction and the arm start pose for robustness.
