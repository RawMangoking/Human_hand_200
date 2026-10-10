#!/usr/bin/env python3
"""
experiment.py - the experiments of the research question, end to end.

  Q1  Does domain-randomized training generalize better to UNSEEN doors than training on one fixed door?
      -> train --condition fixed  vs  --condition random, evaluate both on the same test doors
  Q2  Do modular policies with reduced DOF learn better than controlling every joint / one policy for everything?
      -> --action-mode synergy vs full;  task handle/door (modular) vs task full (one policy), chain the modular ones

COMMANDS
  python experiment.py plan                                    # prints the whole experiment matrix as commands
  python experiment.py train --task door --condition random --algo ars --seed 0
  python experiment.py train --task door --condition fixed  --algo sac --steps 300000 --seed 0
  python experiment.py eval  --run runs/door_ars_random_synergy_s0 --doors all
  python experiment.py chain --reach runs/reach_... --handle runs/handle_... --door runs/door_... --doors unseen
  python experiment.py report                                  # results.md, results.csv, plots in results/

Algorithms: ars (numpy only, see ars.py), sac / ppo (stable-baselines3:  python -m pip install stable-baselines3)
Test doors (door_env.door_set): nominal | in_dist (10 doors inside the training ranges) | unseen (8 doors outside)
"""
import argparse
import csv
import glob
import json
import math
import os
import time

import numpy as np

import ars as ars_mod
from door_env import DEFAULT_DOOR, DEFAULT_HAND, DoorEnv, door_set

DT = 0.04                       # s per control step (25 Hz)


# --------------------------------------------------------------------------- policies
def load_policy(run_dir):
    """-> (policy with .predict(obs), config dict)"""
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    if cfg["algo"] == "ars":
        best = os.path.join(run_dir, "best.npz")
        return ars_mod.LinearPolicy.load(best if os.path.exists(best) else os.path.join(run_dir, "policy.npz")), cfg
    from stable_baselines3 import PPO, SAC
    cls = {"sac": SAC, "ppo": PPO}[cfg["algo"]]
    return cls.load(os.path.join(run_dir, "model.zip"), device="cpu"), cfg


def run_name(a):
    return f"{a.task}_{a.algo}_{a.condition}_{a.action_mode}_s{a.seed}"


def make_env(cfg_or_args, **over):
    g = (lambda k, d=None: cfg_or_args.get(k, d)) if isinstance(cfg_or_args, dict) else \
        (lambda k, d=None: getattr(cfg_or_args, k, d))
    kw = dict(task=g("task"), hand_path=g("hand", DEFAULT_HAND), door_path=g("door", DEFAULT_DOOR),
              action_mode=g("action_mode", "synergy"), obs_noise=g("obs_noise", 0.0),
              randomize_physics=(g("condition", "fixed") == "random"))
    kw.update(over)
    return DoorEnv(**kw)


# --------------------------------------------------------------------------- train
def cmd_train(a):
    a.out = a.out or os.path.join(a.runs, run_name(a))
    os.makedirs(a.out, exist_ok=True)
    print(f"[train] {run_name(a)} -> {a.out}")
    if a.algo == "ars":
        a.iters = a.iters or 100
        a.dirs, a.top, a.step, a.noise = a.dirs, a.top, a.lr, a.noise
        a.eval_every, a.eval_episodes = 5, 10
        a.workers = a.workers or max(1, min(16, (os.cpu_count() or 2) - 1))
        ars_mod.train(a)
        return
    try:
        from stable_baselines3 import PPO, SAC
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.monitor import Monitor
    except ImportError:
        raise SystemExit("needs stable-baselines3:  python -m pip install stable-baselines3   (or use --algo ars)")
    json.dump(dict(vars(a)), open(os.path.join(a.out, "config.json"), "w"), indent=2)
    env = Monitor(make_env(a), os.path.join(a.out, "monitor.csv"), info_keywords=("success",))
    eval_env = make_env(a, randomize_physics=False)

    class Progress(BaseCallback):
        def __init__(self):
            super().__init__()
            self.f = open(os.path.join(a.out, "progress.csv"), "w", newline="")
            self.w = csv.writer(self.f)
            self.w.writerow(["iteration", "env_steps", "mean_return", "eval_return", "eval_success", "seconds"])
            self.t0, self.k = time.time(), 0

        def _on_step(self):
            if self.num_timesteps % a.eval_every_steps == 0:
                self.k += 1
                rets, succ = [], []
                for i in range(10):
                    obs, _ = eval_env.reset(seed=10_000_000 + i)
                    done, ret = False, 0.0
                    while not done:
                        act, _ = self.model.predict(obs, deterministic=True)
                        obs, r, term, trunc, info = eval_env.step(act)
                        ret += r
                        done = term or trunc
                    rets.append(ret)
                    succ.append(info["success"])
                self.w.writerow([self.k, self.num_timesteps, "", f"{np.mean(rets):.4f}", f"{np.mean(succ):.3f}",
                                 f"{time.time() - self.t0:.1f}"])
                self.f.flush()
                print(f"[{a.algo}] steps {self.num_timesteps:8d}  eval return {np.mean(rets):8.2f}  "
                      f"eval success {np.mean(succ) * 100:5.1f} %")
            return True

    algo = {"sac": SAC, "ppo": PPO}[a.algo]
    kw = dict(verbose=0, seed=a.seed, device="auto")
    if a.algo == "sac":
        kw.update(learning_starts=2000, batch_size=256)
    model = algo("MlpPolicy", env, **kw)
    model.learn(total_timesteps=a.steps, callback=Progress())
    model.save(os.path.join(a.out, "model.zip"))
    print(f"[train] saved {a.out}/model.zip")


# --------------------------------------------------------------------------- evaluate
def evaluate(policy, cfg, doors, episodes, seed0=100_000):
    """-> list of per-door metric dicts"""
    env = make_env(cfg, randomize_physics=False)
    rows = []
    for label, phys in doors:
        stats = dict(success=[], time=[], door_deg=[], handle=[], dist=[], ret=[])
        for i in range(episodes):
            obs, _ = env.reset(seed=seed0 + i, options={"physics": phys})
            done, ret = False, 0.0
            while not done:
                act, _ = policy.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(act)
                ret += r
                done = term or trunc
            stats["success"].append(info["success"])
            stats["time"].append(env.steps * DT if info["success"] else np.nan)
            stats["door_deg"].append(info["door_deg"])
            stats["handle"].append(info["handle_frac"])
            stats["dist"].append(info["palm_to_handle"])
            stats["ret"].append(ret)
        rows.append(dict(door=label, physics=json.dumps(phys), episodes=episodes,
                         success_rate=float(np.mean(stats["success"])),
                         time_s=float(np.nanmean(stats["time"])) if any(stats["success"]) else float("nan"),
                         final_door_deg=float(np.mean(stats["door_deg"])),
                         handle_frac=float(np.mean(stats["handle"])),
                         palm_to_handle_mm=1000 * float(np.mean(stats["dist"])),
                         mean_return=float(np.mean(stats["ret"]))))
    env.close()
    return rows


def write_rows(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def print_rows(title, rows):
    print(f"\n{title}")
    print(f"  {'door':<18}{'success':>9}{'time s':>8}{'door deg':>10}{'handle %':>10}{'palm mm':>9}")
    for r in rows:
        print(f"  {r['door']:<18}{r['success_rate'] * 100:8.0f}%{r['time_s']:8.1f}{r['final_door_deg']:10.1f}"
              f"{r['handle_frac'] * 100:10.0f}{r['palm_to_handle_mm']:9.0f}")
    print(f"  {'MEAN':<18}{np.mean([r['success_rate'] for r in rows]) * 100:8.0f}%")


def cmd_eval(a):
    for run in a.run:
        pol, cfg = load_policy(run)
        for set_name in (["nominal", "in_dist", "unseen"] if a.doors == "all" else [a.doors]):
            rows = evaluate(pol, cfg, door_set(set_name), a.episodes)
            write_rows(os.path.join(run, f"eval_{set_name}.csv"), rows)
            print_rows(f"[eval] {os.path.basename(run)} on '{set_name}' doors ({a.episodes} episodes each)", rows)


# --------------------------------------------------------------------------- chain the three modular policies
def cmd_chain(a):
    pols = {t: load_policy(getattr(a, t)) for t in ("reach", "handle", "door")}
    cfg = dict(pols["reach"][1])
    env = make_env(cfg, task="reach", randomize_physics=False)
    limits = {"reach": 250, "handle": 250, "door": 250}
    out_rows = []
    for set_name in (["nominal", "in_dist", "unseen"] if a.doors == "all" else [a.doors]):
        for label, phys in door_set(set_name):
            per = dict(reach=0, handle=0, door=0, total=0, time=[])
            for i in range(a.episodes):
                obs, _ = env.reset(seed=200_000 + i, options={"physics": phys})
                t_all, ok = 0, True
                for stage in ("reach", "handle", "door"):
                    if stage != "reach":
                        obs = env.switch_task(stage)
                    pol = pols[stage][0]
                    if pols[stage][1].get("action_mode", "synergy") != env.action_mode:
                        raise SystemExit("all three policies must use the same --action-mode")
                    done, info = False, {}
                    while not done:
                        act, _ = pol.predict(obs, deterministic=True)
                        obs, r, term, trunc, info = env.step(act)
                        done = term or env.steps >= limits[stage]
                    t_all += env.steps
                    if not info.get("success"):
                        ok = False
                        break
                    per[stage] += 1
                if ok:
                    per["total"] += 1
                    per["time"].append(t_all * DT)
            row = dict(set=set_name, door=label, episodes=a.episodes,
                       reach=per["reach"] / a.episodes, handle=per["handle"] / a.episodes,
                       door_open=per["door"] / a.episodes, end_to_end=per["total"] / a.episodes,
                       time_s=float(np.mean(per["time"])) if per["time"] else float("nan"))
            out_rows.append(row)
            print(f"  {set_name:<8} {label:<18} reach {row['reach'] * 100:4.0f}%  handle {row['handle'] * 100:4.0f}%"
                  f"  door {row['door_open'] * 100:4.0f}%  END-TO-END {row['end_to_end'] * 100:4.0f}%  "
                  f"time {row['time_s']:.1f} s")
    os.makedirs(a.out, exist_ok=True)
    write_rows(os.path.join(a.out, "chain_eval.csv"), out_rows)
    print(f"[chain] written {a.out}/chain_eval.csv")


# --------------------------------------------------------------------------- report
def cmd_report(a):
    runs = sorted(d for d in glob.glob(os.path.join(a.runs, "*")) if os.path.exists(os.path.join(d, "config.json")))
    if not runs:
        raise SystemExit(f"no runs in {a.runs}/")
    os.makedirs(a.out, exist_ok=True)
    groups = {}
    for r in runs:
        c = json.load(open(os.path.join(r, "config.json")))
        key = (c["task"], c["algo"], c.get("condition", "fixed"), c.get("action_mode", "synergy"))
        g = groups.setdefault(key, dict(runs=[], sets={}))
        g["runs"].append(r)
        for s in ("nominal", "in_dist", "unseen"):
            p = os.path.join(r, f"eval_{s}.csv")
            if os.path.exists(p):
                rows = list(csv.DictReader(open(p)))
                g["sets"].setdefault(s, []).append(np.mean([float(x["success_rate"]) for x in rows]))
    lines = ["| task | algo | training | actions | seeds | nominal | in-distribution | UNSEEN |",
             "|---|---|---|---|:-:|:-:|:-:|:-:|"]
    table = []
    fmt = lambda v: f"{np.mean(v) * 100:.0f} ± {np.std(v) * 100:.0f} %" if v else "–"
    for key, g in sorted(groups.items()):
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {len(g['runs'])} | "
                     f"{fmt(g['sets'].get('nominal'))} | {fmt(g['sets'].get('in_dist'))} | "
                     f"{fmt(g['sets'].get('unseen'))} |")
        table.append(dict(task=key[0], algo=key[1], condition=key[2], action_mode=key[3], seeds=len(g["runs"]),
                          **{s: float(np.mean(v)) for s, v in g["sets"].items()}))
    md = "# Results (success rate, mean ± sd over seeds)\n\n" + "\n".join(lines) + "\n"
    open(os.path.join(a.out, "results.md"), "w").write(md)
    if table:
        keys = sorted({k for t in table for k in t})
        with open(os.path.join(a.out, "results.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(table)
    print(md)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(python -m pip install matplotlib for plots)")
        return
    # learning curves
    for task in sorted({k[0] for k in groups}):
        fig, ax = plt.subplots(figsize=(7, 4))
        for key, g in sorted(groups.items()):
            if key[0] != task:
                continue
            curves = []
            for r in g["runs"]:
                p = os.path.join(r, "progress.csv")
                if os.path.exists(p):
                    rows = list(csv.DictReader(open(p)))
                    curves.append(([int(x["env_steps"]) for x in rows], [float(x["eval_success"]) for x in rows]))
            if not curves:
                continue
            n = min(len(c[0]) for c in curves)
            xs = np.array(curves[0][0][:n])
            ys = np.array([c[1][:n] for c in curves])
            lab = f"{key[1]} / {key[2]} / {key[3]}"
            ax.plot(xs, ys.mean(0) * 100, label=lab)
            if len(curves) > 1:
                ax.fill_between(xs, (ys.mean(0) - ys.std(0)) * 100, (ys.mean(0) + ys.std(0)) * 100, alpha=0.2)
        ax.set_xlabel("environment steps")
        ax.set_ylabel("eval success (%)")
        ax.set_title(f"Learning curves - task '{task}'")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"learning_{task}.png"), dpi=150)
        plt.close(fig)
    # generalization bars (unseen doors, per condition)
    for task in sorted({k[0] for k in groups}):
        per = {}
        for key, g in groups.items():
            if key[0] != task:
                continue
            for r in g["runs"]:
                p = os.path.join(r, "eval_unseen.csv")
                if os.path.exists(p):
                    for x in csv.DictReader(open(p)):
                        per.setdefault(f"{key[2]}/{key[3]}", {}).setdefault(x["door"], []).append(
                            float(x["success_rate"]))
        if not per:
            continue
        doors = sorted({d for v in per.values() for d in v})
        fig, ax = plt.subplots(figsize=(9, 4))
        w = 0.8 / len(per)
        for i, (cond, v) in enumerate(sorted(per.items())):
            ax.bar(np.arange(len(doors)) + i * w, [np.mean(v.get(d, [0])) * 100 for d in doors], w, label=cond)
        ax.set_xticks(np.arange(len(doors)) + 0.4 - w / 2)
        ax.set_xticklabels(doors, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("success on unseen door (%)")
        ax.set_title(f"Generalization to unseen doors - task '{task}'")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"unseen_{task}.png"), dpi=150)
        plt.close(fig)
    print(f"[report] {a.out}/results.md, results.csv, learning_*.png, unseen_*.png")


# --------------------------------------------------------------------------- plan
def cmd_plan(a):
    seeds = range(a.seeds)
    print("# Q1 - fixed vs domain-randomized training (modular policies, synergy actions)")
    for task in ("reach", "handle", "door"):
        for cond in ("fixed", "random"):
            for s in seeds:
                print(f"python experiment.py train --task {task} --condition {cond} --algo {a.algo} --seed {s}")
    print("\n# Q2a - reduced (synergy) vs full-joint actions (handle and door, randomized training)")
    for task in ("handle", "door"):
        for s in seeds:
            print(f"python experiment.py train --task {task} --condition random --action-mode full "
                  f"--algo {a.algo} --seed {s}")
    print("\n# Q2b - one policy for everything (task full) vs the chained modular policies")
    for s in seeds:
        print(f"python experiment.py train --task full --condition random --algo {a.algo} --seed {s}")
    print("\n# evaluate every run on every test-door set, then build the tables / plots")
    print("Get-ChildItem runs -Directory | ForEach-Object { python experiment.py eval --run $_.FullName --doors all }")
    print("python experiment.py chain --reach runs/reach_X --handle runs/handle_X --door runs/door_X --doors all "
          "--out runs/chain_X")
    print("python experiment.py report")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    ars_mod.add_args(t)
    t.add_argument("--algo", choices=["ars", "sac", "ppo"], default="ars")
    t.add_argument("--steps", type=int, default=300_000, help="sac/ppo environment steps")
    t.add_argument("--eval-every-steps", type=int, default=10_000)
    t.add_argument("--iters", type=int, default=None, help="ars iterations (default 100)")
    t.add_argument("--dirs", type=int, default=16)
    t.add_argument("--top", type=int, default=8)
    t.add_argument("--lr", type=float, default=0.02, help="ars step size")
    t.add_argument("--noise", type=float, default=0.03)
    t.add_argument("--workers", type=int, default=None)
    t.add_argument("--runs", default="runs")
    t.add_argument("--out", default=None)

    e = sub.add_parser("eval")
    e.add_argument("--run", nargs="+", required=True)
    e.add_argument("--doors", choices=["nominal", "in_dist", "unseen", "all"], default="all")
    e.add_argument("--episodes", type=int, default=10)

    c = sub.add_parser("chain")
    for k in ("reach", "handle", "door"):
        c.add_argument(f"--{k}", required=True, help=f"run directory of the {k} policy")
    c.add_argument("--doors", choices=["nominal", "in_dist", "unseen", "all"], default="all")
    c.add_argument("--episodes", type=int, default=10)
    c.add_argument("--out", default=os.path.join("runs", "chain"))

    r = sub.add_parser("report")
    r.add_argument("--runs", default="runs")
    r.add_argument("--out", default="results")

    p = sub.add_parser("plan")
    p.add_argument("--algo", choices=["ars", "sac", "ppo"], default="sac")
    p.add_argument("--seeds", type=int, default=3)

    a = ap.parse_args()
    dict(train=cmd_train, eval=cmd_eval, chain=cmd_chain, report=cmd_report, plan=cmd_plan)[a.cmd](a)


if __name__ == "__main__":
    main()
