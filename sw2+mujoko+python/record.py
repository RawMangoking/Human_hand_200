#!/usr/bin/env python3
"""
record.py - save videos (MP4, or GIF if ffmpeg is missing) of the hand + door environment for reviews / reports.
An overlay shows the door, stage, door angle and handle turn.

  python record.py --task full --policy scripted --doors unseen              # hand-written policy, unseen doors
  python record.py --task reach --policy runs/reach_ars_random_synergy_s0 --doors nominal
  python record.py --chain runs/reach_X runs/handle_X runs/door_X --doors unseen     # the 3 modular policies
  python record.py --task full --policy scripted --doors nominal --width 1280 --height 720

Needs:  python -m pip install imageio imageio-ffmpeg pillow
Videos go to videos/<name>/<door>_ep<k>.mp4
"""
import argparse
import os

import numpy as np

from door_env import DEFAULT_DOOR, DEFAULT_HAND, DoorEnv, door_set, scripted_action


class ScriptedPolicy:
    def __init__(self, env):
        self.env = env

    def predict(self, obs, deterministic=True):
        return scripted_action(self.env), None


def overlay(frame, lines):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return frame
    img = Image.fromarray(frame)
    dr = ImageDraw.Draw(img)
    h = 16
    dr.rectangle([0, 0, 330, 8 + h * len(lines)], fill=(0, 0, 0))
    for i, t in enumerate(lines):
        dr.text((8, 4 + h * i), t, fill=(255, 255, 255))
    return np.array(img)


def save(frames, path, fps):
    import imageio
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        imageio.mimsave(path, frames, fps=fps, macro_block_size=1)
        return path
    except Exception:                                   # no ffmpeg -> GIF
        gif = os.path.splitext(path)[0] + ".gif"
        imageio.mimsave(gif, frames[::2], duration=2.0 / fps)
        return gif


def make_renderer(env, width, height, azimuth, elevation, distance):
    import mujoco
    r = mujoco.Renderer(env.model, height, width)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation

    def grab():
        cam.lookat[:] = env.geo["door"]["grasp"]
        r.update_scene(env.data, cam)
        return r.render()
    return grab


def rollout(env, stages, label, grab, max_steps=None):
    """stages: [(task, policy)] - one stage = a single policy; several = chained with switch_task.
    -> (frames, info)"""
    frames, info = [], {}
    for k, (task, pol) in enumerate(stages):
        if k > 0:
            env.switch_task(task)
        obs = env._obs()
        done = False
        while not done:
            act, _ = pol.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(act)
            done = term or trunc or (max_steps and env.steps >= max_steps)
            frames.append(overlay(grab(), [
                f"door: {label}",
                f"stage: {env.task}" + (f" ({env._stage})" if env.task == "full" else ""),
                f"door {info['door_deg']:5.1f} deg   handle {info['handle_frac'] * 100:4.0f} %",
                f"palm-handle {info['palm_to_handle'] * 1000:4.0f} mm   {'SUCCESS' if info['success'] else ''}"]))
        if not info.get("success"):
            break
    return frames, info


def load(run, env):
    if run == "scripted":
        return ScriptedPolicy(env)
    from experiment import load_policy
    return load_policy(run)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=list(DoorEnv.TASKS), default="full")
    ap.add_argument("--policy", default="scripted", help="'scripted' or a run directory")
    ap.add_argument("--chain", nargs=3, metavar=("REACH", "HANDLE", "DOOR"),
                    help="three run directories (or 'scripted') chained reach -> handle -> door")
    ap.add_argument("--doors", choices=["nominal", "in_dist", "unseen", "all"], default="nominal")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--action-mode", choices=["synergy", "full"], default="synergy")
    ap.add_argument("--hand", default=DEFAULT_HAND)
    ap.add_argument("--door", default=DEFAULT_DOOR)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--azimuth", type=float, default=135)
    ap.add_argument("--elevation", type=float, default=-15)
    ap.add_argument("--distance", type=float, default=1.6)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--out", default="videos")
    args = ap.parse_args()

    task = "reach" if args.chain else args.task
    env = DoorEnv(task=task, hand_path=args.hand, door_path=args.door, action_mode=args.action_mode)
    grab = make_renderer(env, args.width, args.height, args.azimuth, args.elevation, args.distance)
    if args.chain:
        stages = [(t, load(r, env)) for t, r in zip(("reach", "handle", "door"), args.chain)]
        name = "chain"
    else:
        stages = [(task, load(args.policy, env))]
        name = f"{task}_{'scripted' if args.policy == 'scripted' else os.path.basename(args.policy.rstrip('/'))}"
    for label, phys in door_set(args.doors):
        for ep in range(args.episodes):
            env.reset(seed=300 + ep, options={"physics": phys})
            frames, info = rollout(env, stages, label, grab)
            path = save(frames, os.path.join(args.out, name, f"{label}_ep{ep}.mp4"), args.fps)
            print(f"[record] {label:<16} ep {ep}: {'SUCCESS' if info.get('success') else 'fail   '} "
                  f"door {info.get('door_deg', 0):5.1f} deg, {len(frames)} frames -> {path}")


if __name__ == "__main__":
    main()
