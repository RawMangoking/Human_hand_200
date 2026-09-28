#!/usr/bin/env python3
"""
train_policy.py - train one of the three door tasks with stable-baselines3 (SAC).

  python -m pip install stable-baselines3 tensorboard
  python train_policy.py --task reach --steps 200000          # trains, saves policies/reach_sac.zip
  python train_policy.py --task reach --play                  # watch the trained policy
  tensorboard --logdir runs                                   # learning curves

Start with 'reach' (quickest to learn), then 'handle', then 'door'.
"""
import argparse
import os
import time

from door_env import DoorEnv, DEFAULT_DOOR, DEFAULT_HAND


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=list(DoorEnv.TASKS), default="reach")
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--play", action="store_true", help="load the saved policy and watch it")
    ap.add_argument("--hand", default=DEFAULT_HAND)
    ap.add_argument("--door", default=DEFAULT_DOOR)
    args = ap.parse_args()
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.monitor import Monitor
    except ImportError:
        raise SystemExit("needs stable-baselines3:  python -m pip install stable-baselines3 tensorboard")

    os.makedirs("policies", exist_ok=True)
    path = os.path.join("policies", f"{args.task}_sac.zip")

    if args.play:
        env = DoorEnv(task=args.task, hand_path=args.hand, door_path=args.door, render_mode="human")
        model = SAC.load(path)
        for ep in range(5):
            obs, info = env.reset(seed=100 + ep)
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(action)
                done = term or trunc
                time.sleep(1 / 25)
            print(f"[play {ep}] success {info['success']}  palm->handle {info['palm_to_handle'] * 1000:.0f} mm  "
                  f"handle {info['handle_frac'] * 100:.0f} %  door {info['door_deg']:.1f} deg")
        env.close()
        return

    env = Monitor(DoorEnv(task=args.task, hand_path=args.hand, door_path=args.door))
    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="runs", learning_starts=2000, batch_size=256)
    model.learn(total_timesteps=args.steps, tb_log_name=f"{args.task}_sac")
    model.save(path)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
