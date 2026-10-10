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
    tag = f"_{a.tag}" if getattr(a, "tag", None) else ""
    return f"{a.task}_{a.algo}_{a.condition}_{a.action_mode}{tag}_s{a.seed}"


def make_env(cfg_or_args, **over):
    g = (lambda k, d=None: cfg_or_args.get(k, d)) if isinstance(cfg_or_args, dict) else \
        (lambda k, d=None: getattr(cfg_or_args, k, d))
    kw = dict(task=g("task"), hand_path=g("hand", DEFAULT_HAND), door_path=g("door", DEFAULT_DOOR),
              action_mode=g("action_mode", "synergy"), obs_noise=g("obs_noise", 0.0),
              randomize_physics=(g("condition", "fixed") == "random"))
    starts = ars_mod.load_starts(g("starts", None))
    if starts:
        kw.update(start_states=starts, start_mix=g("start_mix", 0.5))
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
    eval_env = make_env(a, randomize_physics=False, start_states=None)

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
    env = make_env(cfg, randomize_physics=False, start_states=None)
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


# --------------------------------------------------------------------------- start-state pools
def cmd_starts(a):
    """Run a policy (trained, or 'scripted') and keep the simulation state every time it SUCCEEDS:
    reach -> pool to train 'handle' from, handle -> pool to train 'door' from."""
    import pickle
    from door_env import pick_scripted_variant, scripted_action
    env = DoorEnv(task=a.task, hand_path=a.hand, door_path=a.door, action_mode=a.action_mode,
                  randomize_physics=(a.condition == "random"))
    if a.policy == "scripted":
        sc = pick_scripted_variant(env)
        print(f"[starts] hand-written policy, strategy '{env.scripted_variant}' ({sc})")
        act_fn = lambda obs: scripted_action(env)
    else:
        pol, cfg = load_policy(a.policy)
        act_fn = lambda obs: pol.predict(obs, deterministic=True)[0]
    states, t0 = [], time.time()
    for ep in range(a.episodes):
        obs, _ = env.reset(seed=30_000_000 + ep)
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(act_fn(obs))
            done = term or trunc
        if info["success"]:
            states.append(env.get_state())
        if (ep + 1) % 20 == 0:
            print(f"[starts] {ep + 1}/{a.episodes} episodes, {len(states)} successful end states")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump(states, f)
    nxt = {"reach": "handle", "handle": "door"}.get(a.task, "?")
    print(f"[starts] {len(states)}/{a.episodes} successes ({time.time() - t0:.0f} s) -> {a.out}\n"
          f"         train the next stage from them:  python experiment.py train --task {nxt} --starts {a.out} ...")


# --------------------------------------------------------------------------- chain the three modular policies
def cmd_chain(a):
    """Run reach -> handle -> door on one door with the three trained policies. With recovery (default):
    if the hand loses the lever during 'handle' it goes back to 'reach'; if the latch re-engages during 'door'
    (the handle slipped back) it goes back to 'handle'."""
    pols = {t: load_policy(getattr(a, t)) for t in ("reach", "handle", "door")}
    modes = {p[1].get("action_mode", "synergy") for p in pols.values()}
    if len(modes) > 1:
        raise SystemExit("all three policies must use the same --action-mode")
    cfg = dict(pols["reach"][1])
    env = make_env(cfg, task="reach", randomize_physics=False, start_states=None)
    out_rows = []
    for set_name in (["nominal", "in_dist", "unseen"] if a.doors == "all" else [a.doors]):
        for label, phys in door_set(set_name):
            stats = dict(reach=0, handle=0, door=0, total=0, time=[], recover=[])
            for i in range(a.episodes):
                obs, _ = env.reset(seed=200_000 + i, options={"physics": phys})
                stage, steps_total, in_stage, recov, done_all = "reach", 0, 0, 0, False
                reached = set()
                while steps_total < a.max_steps:
                    act, _ = pols[stage][0].predict(obs, deterministic=True)
                    obs, r, term, trunc, info = env.step(act)
                    steps_total += 1
                    in_stage += 1
                    nxt = None
                    if info["success"]:
                        reached.add(stage)
                        if stage == "door":
                            done_all = True
                            break
                        nxt = {"reach": "handle", "handle": "door"}[stage]
                    elif not a.no_recovery and recov < a.max_recoveries:
                        if stage == "handle" and info["palm_to_handle"] > 0.12 and in_stage > 10:
                            nxt, recov = "reach", recov + 1               # lost the lever
                        elif stage == "door" and env._latched:
                            nxt, recov = "handle", recov + 1              # handle slipped back, latch re-engaged
                    if nxt is None and in_stage >= a.stage_steps:
                        break                                             # this stage timed out
                    if nxt:
                        obs, stage, in_stage = env.switch_task(nxt), nxt, 0
                for st in reached:
                    stats[st] += 1
                if done_all:
                    stats["total"] += 1
                    stats["time"].append(steps_total * DT)
                stats["recover"].append(recov)
            n = a.episodes
            row = dict(set=set_name, door=label, episodes=n, reach=stats["reach"] / n, handle=stats["handle"] / n,
                       door_open=stats["door"] / n, end_to_end=stats["total"] / n,
                       time_s=float(np.mean(stats["time"])) if stats["time"] else float("nan"),
                       recoveries=float(np.mean(stats["recover"])))
            out_rows.append(row)
            print(f"  {set_name:<8} {label:<16} reach {row['reach'] * 100:4.0f}%  handle {row['handle'] * 100:4.0f}%"
                  f"  door {row['door_open'] * 100:4.0f}%  END-TO-END {row['end_to_end'] * 100:4.0f}%  "
                  f"time {row['time_s']:5.1f} s  recoveries {row['recoveries']:.1f}")
    os.makedirs(a.out, exist_ok=True)
    write_rows(os.path.join(a.out, "chain_eval.csv"), out_rows)
    print(f"[chain] mean end-to-end {np.mean([r['end_to_end'] for r in out_rows]) * 100:.0f} %  -> "
          f"{a.out}/chain_eval.csv")


def _chain_runs(runs_dir):
    """{condition: {seed: rows}} from runs/chain_<algo>_<condition>_s<seed>/chain_eval.csv"""
    out = {}
    for c in glob.glob(os.path.join(runs_dir, "chain_*", "chain_eval.csv")):
        parts = os.path.basename(os.path.dirname(c)).split("_")
        if len(parts) < 4:
            continue
        cond, seed = parts[2], parts[3]
        out.setdefault(cond, {})[seed] = list(csv.DictReader(open(c)))
    return out


def _bootstrap_diff(a_by_door, b_by_door, n=10000, seed=0):
    """95 % CI of mean(b - a) over doors (paired: same test doors), resampling doors."""
    doors = sorted(set(a_by_door) & set(b_by_door))
    if len(doors) < 2:
        return float("nan"), float("nan"), float("nan")
    diff = np.array([b_by_door[d] - a_by_door[d] for d in doors])
    rng = np.random.default_rng(seed)
    boots = diff[rng.integers(0, len(diff), (n, len(diff)))].mean(1)
    return float(diff.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def research_summary(runs_dir, groups):
    md = "\n# Research questions\n\n"
    ch = _chain_runs(runs_dir)
    # ---- Q1: fixed vs randomized training, three chained policies, end-to-end success
    md += "## Q1 - fixed vs domain-randomized training (three chained policies, end-to-end success)\n\n"
    if ch:
        md += "| training | seeds | nominal | in-distribution | **unseen** |\n|---|:-:|:-:|:-:|:-:|\n"
        per_door = {}
        for cond in sorted(ch):
            cells = []
            for sname in ("nominal", "in_dist", "unseen"):
                vals = []
                for seed, rows in ch[cond].items():
                    rs = [r for r in rows if r["set"] == sname]
                    if rs:
                        vals.append(np.mean([float(r["end_to_end"]) for r in rs]))
                    for r in rs:
                        if sname == "unseen":
                            per_door.setdefault(cond, {}).setdefault(r["door"], []).append(float(r["end_to_end"]))
                cells.append(f"{np.mean(vals) * 100:.0f} ± {np.std(vals) * 100:.0f} %" if vals else "-")
            md += f"| {cond} | {len(ch[cond])} | " + " | ".join(cells) + " |\n"
        if "fixed" in per_door and "random" in per_door:
            fa = {d: np.mean(v) for d, v in per_door["fixed"].items()}
            ra = {d: np.mean(v) for d, v in per_door["random"].items()}
            m, lo, hi = _bootstrap_diff(fa, ra)
            verdict = ("randomized training generalizes BETTER to unseen doors" if lo > 0 else
                       "fixed training generalizes better" if hi < 0 else
                       "no clear difference (the interval includes 0)")
            md += (f"\nUnseen doors, randomized minus fixed: **{m * 100:+.0f} percentage points**, 95 % bootstrap "
                   f"interval [{lo * 100:+.0f}, {hi * 100:+.0f}] over the {len(fa)} doors -> {verdict}.\n")
            md += "\n| unseen door | fixed | randomized |\n|---|:-:|:-:|\n"
            for d in sorted(fa):
                md += f"| {d} | {fa[d] * 100:.0f} % | {ra.get(d, float('nan')) * 100:.0f} % |\n"
    else:
        md += "(no chained runs yet)\n"

    # ---- Q2a: reduced (synergy) vs full-joint actions on the handle stage
    md += ("\n## Q2a - reduced (synergy) actions vs every joint (handle stage, randomized training, "
           "trained FROM SCRATCH - no cloning, so the learning itself is compared)\n\n")
    rows = []
    for mode in ("synergy", "full"):
        g = groups.get(("handle", "ars", "random", f"{mode}/scratch"))
        if not g:
            continue
        to80 = []
        for r in g["runs"]:
            p = os.path.join(r, "progress.csv")
            if os.path.exists(p):
                prog = [x for x in csv.DictReader(open(p)) if x["eval_success"] not in ("", None)]
                hit = [int(x["env_steps"]) for x in prog if float(x["eval_success"]) >= 0.8]
                to80.append(hit[0] if hit else float("nan"))
        un = g["sets"].get("unseen")
        rows.append(f"| {mode} | {len(g['runs'])} | {'-' if not un else f'{np.mean(un) * 100:.0f} %'} | "
                    f"{np.nanmean(to80) / 1000:.0f} k |" if to80 and not np.all(np.isnan(to80)) else
                    f"| {mode} | {len(g['runs'])} | {'-' if not un else f'{np.mean(un) * 100:.0f} %'} | not reached |")
    md += ("| actions | seeds | unseen success | env steps to 80 % eval success |\n|---|:-:|:-:|:-:|\n" + "\n".join(rows)
           + "\n") if rows else "(no handle runs for both action modes yet)\n"

    # ---- Q2b: three chained policies vs one policy for everything
    md += ("\n## Q2b - three chained policies vs one policy for the whole sequence (randomized training; both "
           "cloned from the same hand-written controller with the same budget)\n\n")
    g = groups.get(("full", "ars", "random", "synergy"))
    if g and "random" in ch:
        md += "| approach | nominal | in-distribution | unseen |\n|---|:-:|:-:|:-:|\n"
        cells = []
        for sname in ("nominal", "in_dist", "unseen"):
            vals = [np.mean([float(r["end_to_end"]) for r in rows if r["set"] == sname])
                    for rows in ch["random"].values() if any(r["set"] == sname for r in rows)]
            cells.append(f"{np.mean(vals) * 100:.0f} %" if vals else "-")
        md += "| three chained policies | " + " | ".join(cells) + " |\n"
        md += "| one policy (task full) | " + " | ".join(
            f"{np.mean(g['sets'][s]) * 100:.0f} %" if g["sets"].get(s) else "-"
            for s in ("nominal", "in_dist", "unseen")) + " |\n"
    else:
        md += "(needs the chained runs and the 'full' baseline)\n"
    return md


# --------------------------------------------------------------------------- report
def cmd_report(a):
    runs = sorted(d for d in glob.glob(os.path.join(a.runs, "*")) if os.path.exists(os.path.join(d, "config.json")))
    if not runs:
        raise SystemExit(f"no runs in {a.runs}/")
    os.makedirs(a.out, exist_ok=True)
    groups = {}
    for r in runs:
        c = json.load(open(os.path.join(r, "config.json")))
        key = (c["task"], c["algo"], c.get("condition", "fixed"), c.get("action_mode", "synergy")
               + (f"/{c['tag']}" if c.get("tag") else ""))
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
    # the chained three-policy runs
    chains = sorted(glob.glob(os.path.join(a.runs, "*", "chain_eval.csv")))
    if chains:
        md += ("\n# Chained three-policy runs (reach -> handle -> door)\n\n"
               "| run | door set | doors | reach | handle | door | END-TO-END | time (s) | recoveries |\n"
               "|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|\n")
        for c in chains:
            rows = list(csv.DictReader(open(c)))
            for sname in sorted({r["set"] for r in rows}):
                rs = [r for r in rows if r["set"] == sname]
                f = lambda k: np.mean([float(r[k]) for r in rs])
                tm = [float(r["time_s"]) for r in rs if r["time_s"] not in ("", "nan")]
                md += (f"| {os.path.basename(os.path.dirname(c))} | {sname} | {len(rs)} | {f('reach') * 100:.0f} % | "
                       f"{f('handle') * 100:.0f} % | {f('door_open') * 100:.0f} % | **{f('end_to_end') * 100:.0f} %** | "
                       f"{(np.mean(tm) if tm else float('nan')):.1f} | {f('recoveries'):.1f} |\n")
    md += research_summary(a.runs, groups)
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
    al = a.algo
    bc = " --bc-episodes 40" if al == "ars" else ""
    print("# ===== THE THREE-POLICY PIPELINE (per training condition: fixed / random) =====")
    print("# each stage trains from where the previous one actually ends (start-state pools), with a")
    print("# behaviour-cloning head start from the hand-written controller (ARS)")
    for cond in ("fixed", "random"):
        print(f"\n# --- condition: {cond}")
        for sd in seeds:
            r, h, d = (f"runs/{t}_{al}_{cond}_synergy_s{sd}" for t in ("reach", "handle", "door"))
            print(f"python experiment.py train --task reach  --condition {cond} --algo {al} --seed {sd}{bc}")
            print(f"python experiment.py starts --task reach  --policy {r} --condition {cond} "
                  f"--out starts/after_reach_{cond}_s{sd}.pkl")
            print(f"python experiment.py train --task handle --condition {cond} --algo {al} --seed {sd}{bc} "
                  f"--starts starts/after_reach_{cond}_s{sd}.pkl")
            print(f"python experiment.py starts --task handle --policy {h} --condition {cond} "
                  f"--out starts/after_handle_{cond}_s{sd}.pkl")
            print(f"python experiment.py train --task door   --condition {cond} --algo {al} --seed {sd}{bc} "
                  f"--starts starts/after_handle_{cond}_s{sd}.pkl")
            print(f"python experiment.py chain --reach {r} --handle {h} --door {d} --doors all "
                  f"--out runs/chain_{al}_{cond}_s{sd}")
    print("\n# ===== BASELINES =====")
    print("# reduced (synergy) vs every joint (full) actions")
    for sd in seeds:
        print(f"python experiment.py train --task handle --condition random --action-mode full --algo {al} --seed {sd}")
    print("# one policy for the whole sequence vs the chained three")
    for sd in seeds:
        print(f"python experiment.py train --task full --condition random --algo {al} --seed {sd}{bc}")
    print("\n# ===== EVALUATE + REPORT =====")
    print("Get-ChildItem runs -Directory | Where-Object { $_.Name -notlike 'chain*' } | "
          "ForEach-Object { python experiment.py eval --run $_.FullName --doors all }")
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
    t.add_argument("--tag", default=None, help="extra name part, e.g. 'scratch' for runs without cloning")
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
    c.add_argument("--max-steps", type=int, default=750, help="whole sequence (default 750 = 30 s)")
    c.add_argument("--stage-steps", type=int, default=250, help="one stage (default 250 = 10 s)")
    c.add_argument("--no-recovery", action="store_true", help="never go back to an earlier stage")
    c.add_argument("--max-recoveries", type=int, default=3)
    c.add_argument("--out", default=os.path.join("runs", "chain"))

    st = sub.add_parser("starts", help="start-state pool: end states of successful episodes")
    st.add_argument("--task", choices=["reach", "handle"], required=True)
    st.add_argument("--policy", default="scripted", help="'scripted' or a run directory")
    st.add_argument("--episodes", type=int, default=200)
    st.add_argument("--condition", choices=["fixed", "random"], default="random")
    st.add_argument("--action-mode", choices=["synergy", "full"], default="synergy")
    st.add_argument("--hand", default=DEFAULT_HAND)
    st.add_argument("--door", default=DEFAULT_DOOR)
    st.add_argument("--out", required=True)

    r = sub.add_parser("report")
    r.add_argument("--runs", default="runs")
    r.add_argument("--out", default="results")

    p = sub.add_parser("plan")
    p.add_argument("--algo", choices=["ars", "sac", "ppo"], default="sac")
    p.add_argument("--seeds", type=int, default=3)

    a = ap.parse_args()
    dict(train=cmd_train, eval=cmd_eval, chain=cmd_chain, report=cmd_report, plan=cmd_plan,
         starts=cmd_starts)[a.cmd](a)


if __name__ == "__main__":
    main()
