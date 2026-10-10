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

import pickle

from door_env import DEFAULT_DOOR, DEFAULT_HAND, DoorEnv, pick_scripted_variant, scripted_action


# --------------------------------------------------------------------------- policy
class LinearPolicy:
    """a = clip(W [z, cos(Omega z + b), 1]) with z the normalised observation. n_rff > 0 adds random Fourier
    features (Rahimi & Recht 2007): fixed random nonlinear features, so the policy can be nonlinear in the
    observation while staying linear in W (ARS and least-squares behaviour cloning still apply)."""
    def __init__(self, obs_dim, act_dim, n_rff=0, rff_scale=0.5, seed=0):
        rng = np.random.default_rng(10_000 + seed)
        self.Omega = rng.normal(0, rff_scale, (n_rff, obs_dim))
        self.b = rng.uniform(0, 2 * np.pi, n_rff)
        self.W = np.zeros((act_dim, obs_dim + n_rff + 1))
        self.mean = np.zeros(obs_dim)
        self.var = np.ones(obs_dim)
        self.n = 0

    def features(self, obs):
        z = np.clip((obs - self.mean) / np.sqrt(self.var + 1e-8), -10, 10)
        if len(self.b):
            return np.concatenate([z, np.cos(self.Omega @ z + self.b), [1.0]])
        return np.append(z, 1.0)

    def act(self, obs, W=None):
        return np.clip((self.W if W is None else W) @ self.features(obs), -1, 1).astype(np.float32)

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
        np.savez(path, W=self.W, mean=self.mean, var=self.var, n=self.n, Omega=self.Omega, b=self.b)

    @classmethod
    def load(cls, path):
        z = np.load(path if path.endswith(".npz") else os.path.join(path, "policy.npz"))
        p = cls(len(z["mean"]), z["W"].shape[0])
        p.W, p.mean, p.var, p.n = z["W"], z["mean"], z["var"], int(z["n"])
        if "Omega" in z:
            p.Omega, p.b = z["Omega"], z["b"]
        return p


# --------------------------------------------------------------------------- rollouts (parallel workers)
_ENV = None


def _init_worker(env_kwargs):
    global _ENV
    _ENV = DoorEnv(**env_kwargs)


def _rollout(job):
    W, mean, var, seed, collect, Omega, b = job
    env = _ENV
    pol = LinearPolicy(len(mean), W.shape[0])
    pol.W, pol.mean, pol.var, pol.Omega, pol.b = W, mean, var, Omega, b
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


def load_starts(path):
    if not path:
        return None
    with open(path, "rb") as f:
        states = pickle.load(f)
    return states or None


def env_kwargs_from(args):
    kw = dict(task=args.task, hand_path=args.hand, door_path=args.door, action_mode=args.action_mode,
              randomize_physics=(args.condition == "random"), obs_noise=args.obs_noise)
    starts = load_starts(getattr(args, "starts", None))
    if starts:
        kw.update(start_states=starts, start_mix=getattr(args, "start_mix", 0.5))
    return kw


# --------------------------------------------------------------------------- behaviour cloning (DAPG-style start)
def _fit(pol, O, A, ridge):
    pol.mean, pol.var, pol.n = O.mean(0), O.var(0) + 1e-6, len(O)
    Z = np.array([pol.features(o) for o in O])
    pol.W = np.linalg.solve(Z.T @ Z + ridge * np.eye(Z.shape[1]), Z.T @ A).T


def bc_init(pol, kw, episodes, noise=0.3, seed=0, ridge=1e-2, dagger_rounds=3, log=print):
    """Behaviour cloning from the hand-written controller, with DAgger (Ross et al. 2011):
      round 0 : the controller acts (with noise) and its actions are recorded
      round k : the CLONED policy acts, the controller labels the states it visits, all data is refitted
    so the policy also learns what to do in the states its own mistakes lead to."""
    env = DoorEnv(**kw)
    scores = pick_scripted_variant(env)
    rng = np.random.default_rng(seed)
    O, A = [], []
    per_round = max(1, episodes // (dagger_rounds + 1))
    for rnd in range(dagger_rounds + 1):
        ok = 0
        for ep in range(episodes if rnd == 0 else per_round):
            obs, _ = env.reset(seed=20_000_000 + seed * 10_000 + rnd * 1000 + ep)
            done = False
            while not done:
                a = scripted_action(env)                        # the expert's label for this state
                O.append(obs)
                A.append(a)
                if rnd == 0:
                    a_exec = np.clip(a + rng.normal(0, noise, a.shape), -1, 1)
                else:
                    a_exec = pol.act(obs)                       # DAgger: the cloned policy drives
                obs, _, term, trunc, info = env.step(np.asarray(a_exec, dtype=np.float32))
                done = term or trunc
            ok += info["success"]
        _fit(pol, np.array(O), np.array(A), ridge)
        n = episodes if rnd == 0 else per_round
        log(f"[bc] round {rnd}: {'demonstrations' if rnd == 0 else 'cloned policy drove'} {ok}/{n} succeeded, "
            f"{len(O)} labelled samples")
    env.close()
    log(f"[bc] '{max(scores, key=scores.get)}' controller, {dagger_rounds} DAgger rounds -> linear policy fitted")
    return pol


# --------------------------------------------------------------------------- training
def train(args, log=print):
    kw = env_kwargs_from(args)
    probe = DoorEnv(**kw)
    obs_dim, act_dim = probe.observation_space.shape[0], probe.action_space.shape[0]
    probe.close()
    pol = LinearPolicy(obs_dim, act_dim, n_rff=getattr(args, "rff", 0), seed=args.seed)
    if getattr(args, "bc_episodes", 0):
        bc_init(pol, kw, args.bc_episodes, getattr(args, "bc_noise", 0.3), args.seed,
                dagger_rounds=getattr(args, "dagger", 3), log=log)
    rng = np.random.default_rng(args.seed)
    noise, step = args.noise, args.step
    if getattr(args, "bc_episodes", 0):                     # fine-tuning a good policy: small, careful steps
        noise *= getattr(args, "finetune_scale", 0.25)
        step *= getattr(args, "finetune_scale", 0.25)
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
    ev = run([(pol.W, pol.mean, pol.var, 10_000_000 + i, False, pol.Omega, pol.b) for i in range(args.eval_episodes)])
    ev_ret, ev_succ = float(np.mean([x[0] for x in ev])), float(np.mean([x[2] for x in ev]))
    wr.writerow([0, 0, "", f"{ev_ret:.4f}", f"{ev_succ:.3f}", "0"])
    log(f"[ars] it    0  (start{' = behaviour cloning' if getattr(args, 'bc_episodes', 0) else ''})  "
        f"eval return {ev_ret:8.2f}  eval success {ev_succ * 100:5.1f} %")
    pol.save(os.path.join(args.out, "best.npz"))
    best = ev_ret
    for it in range(1, args.iters + 1):
        deltas = rng.standard_normal((args.dirs,) + pol.W.shape)
        jobs = []
        for k in range(args.dirs):
            sd = seed_ctr + k                                   # same start for +delta and -delta
            jobs.append((pol.W + noise * deltas[k], pol.mean, pol.var, sd, True, pol.Omega, pol.b))
            jobs.append((pol.W - noise * deltas[k], pol.mean, pol.var, sd, True, pol.Omega, pol.b))
        seed_ctr += args.dirs
        res = run(jobs)
        r = np.array([x[0] for x in res]).reshape(args.dirs, 2)
        env_steps += sum(x[1] for x in res)
        S = sum(x[3] for x in res)
        SS = sum(x[4] for x in res)
        N = sum(x[5] for x in res)
        top = np.argsort(-np.max(r, axis=1))[:args.top]          # best directions
        sigma = np.std(r[top]) + 1e-8
        upd = np.sum((r[top, 0] - r[top, 1])[:, None, None] * deltas[top], axis=0)
        pol.W = pol.W + step / (args.top * sigma) * upd
        if not getattr(args, "bc_episodes", 0):         # after cloning the normalisation must stay frozen:
            pol.update_stats(S, SS, N)                  # W was fitted to it (and the Fourier features use it)

        if it % args.eval_every == 0 or it == args.iters:
            ev = run([(pol.W, pol.mean, pol.var, 10_000_000 + i, False, pol.Omega, pol.b) for i in range(args.eval_episodes)])
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
    ap.add_argument("--starts", default=None, help="start-state pool (.pkl from 'experiment.py starts'): train "
                                                     "from where the previous policy ended")
    ap.add_argument("--start-mix", type=float, default=0.5, help="share of episodes started from the pool")
    ap.add_argument("--bc-episodes", type=int, default=0,
                    help="behaviour-cloning start from N hand-written demonstrations (DAPG-style), then ARS")
    ap.add_argument("--bc-noise", type=float, default=0.3, help="action noise while collecting demonstrations")
    ap.add_argument("--rff", type=int, default=128,
                    help="random Fourier features (default 128): lets the policy be nonlinear; 0 = purely linear")
    ap.add_argument("--finetune-scale", type=float, default=0.25,
                    help="after behaviour cloning, ARS noise and step are scaled by this (default 0.25)")
    ap.add_argument("--dagger", type=int, default=3, help="DAgger rounds after the first demonstrations (0 = plain BC)")


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
