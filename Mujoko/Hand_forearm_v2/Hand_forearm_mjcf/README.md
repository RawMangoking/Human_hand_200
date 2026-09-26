# Hand_forearm_mjcf

MuJoCo model exported by sw2robot from the SolidWorks assembly `Hand_forearm`.

    mjcf/Hand_forearm.xml   the model
    mjcf/assets/      its binary-STL mesh assets

Load it with:

    import mujoco
    model = mujoco.MjModel.from_xml_path("mjcf/Hand_forearm.xml")
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)   # the "home" keyframe

What the exporter derived from the CAD, rather than guessed:

* joint damping from each joint's own `effort / velocity` (N*m*s/rad): 3.185 on 15 joints
* `home` keyframe with the base at 0 m, the height at which nothing is below the floor
