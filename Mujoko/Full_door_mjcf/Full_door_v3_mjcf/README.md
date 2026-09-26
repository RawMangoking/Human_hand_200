# Full_door_v3_mjcf

MuJoCo model exported by sw2robot from the SolidWorks assembly `Full_door_v3`.

    mjcf/Full_door_v3.xml   the model
    mjcf/assets/      its binary-STL mesh assets

Load it with:

    import mujoco
    model = mujoco.MjModel.from_xml_path("mjcf/Full_door_v3.xml")
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)   # the "home" keyframe

What the exporter derived from the CAD, rather than guessed:

* joint damping from each joint's own `effort / velocity` (N*m*s/rad): 3.185 on 2 joints
* `home` keyframe with the base at 0 m, the height at which nothing is below the floor
