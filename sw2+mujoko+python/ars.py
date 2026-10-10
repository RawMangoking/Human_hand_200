#!/usr/bin/env python3
"""
ars.py - Augmented Random Search (Mania et al., 2018, "Simple random search provides a competitive approach to
reinforcement learning") for the door environment. Numpy only (no PyTorch / GPU needed), runs episodes in
parallel on all CPU cores. A linear policy on normalised observations: a = clip(W [obs_norm, 1], -1, 1).

Useful as (1) a quick check that a task is learnable, (2) a cheap CPU baseline next to SAC/PPO.

  python ars.py --task reach --iters 60 --workers 8 --out runs/ars_reach_fixed_s0
  python ars.py --task door  --condition random --iters 100 --workers 8 --out runs/ars_door_random_s0
  python ars.py --play runs/ars_reach_fixed_s0                    # watch it
"""
import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import time

import numpy as np

from door_env import DEFAULT_DOOR, DEFAULT_HAND, DoorEnv


# --------------------------------------------------------------------------- policy
class LinearPolicy:
    def __init__(self, obs_dim, act_dim):
        self.W = np.zeros((act_dim, obs_dim + 1))
        self.mean = np.zeros(obs_dim)
        self.var = np.ones(obs_dim)
        self.n = 0

    def act(self, obs, W=None):
        z = (obs - self.mean) / np.sqrt(self.var + 1e-8)
        z = np.clip(z, -10, 10)
        return np.clip((self.W if W is None else W) @ np.append(z, 1.0), -1, 1).astype(np.float32)

    predict = lambda self, obs, deterministic=True: (self.act(obs), None)      # stable-baselines-like API

    def update_stats(self, s, ss, n):
        """Merge a batch (sum, sum of squares, count) into the running mean / variance."""
        if n == 0:
            return
        tot = self.n + n
        new_mean = (self.mean * self.n + s) / tot
        new_ex2 = ((self.var + self.mean ** 2) * self.n + ss) / tot
        self.mean, self.var, self.n = new_mean, np.maximum(new_ex2 - new_mean ** 2, 1e-8), tot

    def save(self, path):
        np.savez(path, W=self.W, mean=self.mean, var=self.var, n=self.n)

    @classmethod
    def load(cls, path):
        z = np.load(path if path.endswith(".npz") else os.path.join(path, "policy.npz"))
        p = cls(len(z["mean"]), z["W"].shape[0])
        p.W, p.mean, p.var, p.n = z["W"], z["mean"], z["var"], int(z["n"])
        return p


# --------------------------------------------------------------------------- rollouts (parallel workers)
_ENV = None


def _init_worker(env_kwargs):
    global _ENV
    _ENV = DoorEnv(**env_kwargs)


def _rollout(job):
    W, mean, var, seed, collect = job
    env = _ENV
    pol = LinearPolicy(len(mean), W.shape[0])
    pol.W, pol.mean, pol.var = W, mean, var
    obs, _ = env.reset(seed=int(seed))
    ret, steps, done, info = 0.0, 0, False, {}
    s = np.zeros_like(mean)
    ss = np.zeros_like(mean)
    while not done:
        if collect:
            s += obs
            ss += obs * obs
        obs, r, term, trunc, info = env.step(pol.act(obs))
        ret += r
        steps += 1
        done = term or trunc
    return ret, steps, bool(info.get("success", False)), s, ss, steps if collect else 0


def env_kwargs_from(args):
    return dict(task=args.task, hand_path=args.hand, door_path=args.door, action_mode=args.action_mode,
                randomize_physics=(args.condition == "random"), obs_noise=args.obs_noise)


# --------------------------------------------------------------------------- training
def train(args, log=print):
    kw = env_kwargs_from(args)
    probe = DoorEnv(**kw)
    obs_dim, act_dim = probe.observation_space.shape[0], probe.action_space.shape[0]
    probe.close()
    pol = LinearPolicy(obs_dim, act_dim)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    json.dump(dict(vars(args), algo="ars", obs_dim=obs_dim, act_dim=act_dim),
              open(os.path.join(args.out, "config.json"), "w"), indent=2)
    log_f = open(os.path.join(args.out, "progress.csv"), "w", newline="")
    wr = csv.writer(log_f)
    wr.writerow(["iteration", "env_steps", "mean_return", "eval_return", "eval_success", "seconds"])
    pool = mp.Pool(args.workers, initializer=_init_worker, initargs=(kw,)) if args.workers > 1 else None
    if pool is None:
        _init_worker(kw)
    run = (lambda jobs: pool.map(_rollout, jobs)) if pool else (lambda jobs: [_rollout(j) for j in jobs])

    env_steps, t0, best = 0, time.time(), -np.inf
    seed_ctr = int(args.seed) * 1_000_000
    for it in range(1, args.iters + 1):
        deltas = rng.standard_normal((args.dirs,) + pol.W.shape)
        jobs = []
        for k in range(args.dirs):
            sd = seed_ctr + k                                   # same start for +delta and -delta
            jobs.append((pol.W + args.noise * deltas[k], pol.mean, pol.var, sd, True))
            jobs.append((pol.W - args.noise * deltas[k], pol.mean, pol.var, sd, True))
        seed_ctr += args.dirs
        res = run(jobs)
        r = np.array([x[0] for x in res]).reshape(args.dirs, 2)
        env_steps += sum(x[1] for x in res)
        S = sum(x[3] for x in res)
        SS = sum(x[4] for x in res)
        N = sum(x[5] for x in res)
        top = np.argsort(-np.max(r, axis=1))[:args.top]          # best directions
        sigma = np.std(r[top]) + 1e-8
        step = np.sum((r[top, 0] - r[top, 1])[:, None, None] * deltas[top], axis=0)
        pol.W = pol.W + args.step / (args.top * sigma) * step
        pol.update_stats(S, SS, N)

        if it % args.eval_every == 0 or it == args.iters:
            ev = run([(pol.W, pol.mean, pol.var, 10_000_000 + i, False) for i in range(args.eval_episodes)])
            ev_ret = float(np.mean([x[0] for x in ev]))
            ev_succ = float(np.mean([x[2] for x in ev]))
            wr.writerow([it, env_steps, f"{r.mean():.4f}", f"{ev_ret:.4f}", f"{ev_succ:.3f}",
                         f"{time.time() - t0:.1f}"])
            log_f.flush()
            log(f"[ars] it {it:4d}  steps {env_steps:8d}  return {r.mean():8.2f}  "
                f"eval return {ev_ret:8.2f}  eval success {ev_succ * 100:5.1f} %  ({time.time() - t0:.0f} s)")
            pol.save(os.path.join(args.out, "policy.npz"))
            if ev_ret > best:
                best = ev_ret
                pol.save(os.path.join(args.out, "best.npz"))
    if pool:
        pool.close()
        pool.join()
    log_f.close()
    return pol


def play(path, episodes=5):
    import time as _t
    cfg = json.load(open(os.path.join(path, "config.json")))
    env = DoorEnv(task=cfg["task"], hand_path=cfg["hand"], door_path=cfg["door"], action_mode=cfg["action_mode"],
                  randomize_physics=(cfg["condition"] == "random"), render_mode="human")
    pol = LinearPolicy.load(os.path.join(path, "best.npz"))
    for ep in range(episodes):
        obs, _ = env.reset(seed=500 + ep)
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(pol.act(obs))
            done = term or trunc
            _t.sleep(1 / 25)
        print(f"[play {ep}] success {info['success']}  palm->handle {info['palm_to_handle'] * 1000:.0f} mm  "
              f"handle {info['handle_frac'] * 100:.0f} %  door {info['door_deg']:.1f} deg")
    env.close()


def add_args(ap):
    ap.add_argument("--task", choices=list(DoorEnv.TASKS), default="reach")
    ap.add_argument("--condition", choices=["fixed", "random"], default="fixed",
                    help="fixed = nominal door every episode, random = domain-randomized door physics")
    ap.add_argument("--action-mode", choices=["synergy", "full"], default="synergy")
    ap.add_argument("--obs-noise", type=float, default=0.0)
    ap.add_argument("--hand", default=DEFAULT_HAND)
    ap.add_argument("--door", default=DEFAULT_DOOR)
    ap.add_argument("--seed", type=int, default=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--dirs", type=int, default=16, help="random directions per iteration")
    ap.add_argument("--top", type=int, default=8, help="best directions used for the update")
    ap.add_argument("--step", type=float, default=0.02, help="step size")
    ap.add_argument("--noise", type=float, default=0.03, help="exploration noise")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--out", default=None)
    ap.add_argument("--play", metavar="RUN_DIR")
    args = ap.parse_args()
    if args.play:
        play(args.play)
        return
    args.out = args.out or os.path.join("runs", f"ars_{args.task}_{args.condition}_{args.action_mode}_s{args.seed}")
    train(args)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
