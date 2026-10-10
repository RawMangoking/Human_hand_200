#!/usr/bin/env python3
"""
run_all.py - runs the WHOLE three-policy experiment unattended (e.g. overnight), then evaluates and reports.

  python run_all.py                       # everything: 2 conditions x 3 seeds x 3 stages + baselines
  python run_all.py --dry-run             # only print what it would do
  python run_all.py --seeds 1 --iters 60  # a quicker first pass
  python run_all.py --no-baselines        # only the three-policy pipeline

RESUMABLE: every step checks for its output first (policy / start pool / chain result / evaluation) and is
skipped if it is already there - after a crash, a reboot or Ctrl+C just run the same command again.
A failing step is logged and the rest continues. Everything is logged to run_all.log.

Order per condition (fixed / random) and seed:
  reach -> start pool after reach -> handle (from that pool) -> pool after handle -> door (from it) -> chain
then the baselines - Q2a: handle with synergy vs every-joint actions, both FROM SCRATCH (the DOF effect on
learning); Q2b: one policy for the whole sequence - then evaluation of every run, and the report.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


class Runner:
    def __init__(self, a):
        self.a = a
        self.log_f = open(os.path.join(HERE, "run_all.log"), "a")
        self.done, self.skipped, self.failed = 0, 0, []
        self.t0 = time.time()

    def log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.log_f.write(line + "\n")
        self.log_f.flush()

    def step(self, name, args, output):
        """Run `experiment.py args` unless `output` exists."""
        if output and os.path.exists(os.path.join(HERE, output)):
            self.skipped += 1
            self.log(f"skip  {name}  (already done: {output})")
            return True
        uses_models = args[0] in ("train", "starts")              # the others read paths from the runs
        cmd = [sys.executable, os.path.join(HERE, "experiment.py")] + args + (self.a.model_args if uses_models else [])
        self.log(f"start {name}")
        if self.a.dry_run:
            self.log("      " + " ".join(cmd))
            return True
        t = time.time()
        with open(os.path.join(HERE, "run_all.log"), "a") as out:
            res = subprocess.run(cmd, cwd=HERE, stdout=out, stderr=subprocess.STDOUT)
        if res.returncode == 0 and (not output or os.path.exists(os.path.join(HERE, output))):
            self.done += 1
            self.log(f"done  {name}  ({(time.time() - t) / 60:.1f} min)")
            return True
        self.failed.append(name)
        self.log(f"FAIL  {name}  (exit {res.returncode}) - see run_all.log; continuing")
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--conditions", nargs="+", default=["fixed", "random"], choices=["fixed", "random"])
    ap.add_argument("--iters", type=int, default=150, help="ARS iterations per policy (default 150)")
    ap.add_argument("--dirs", type=int, default=32, help="ARS directions per iteration (default 32)")
    ap.add_argument("--bc", type=int, default=40, help="behaviour-cloning demonstrations (default 40)")
    ap.add_argument("--workers", type=int, default=max(1, min(30, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--pool-episodes", type=int, default=200, help="episodes for each start-state pool")
    ap.add_argument("--eval-episodes", type=int, default=10, help="episodes per test door")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--hand", default=None)
    ap.add_argument("--door", default=None)
    a = ap.parse_args()
    a.model_args = (["--hand", a.hand] if a.hand else []) + (["--door", a.door] if a.door else [])
    r = Runner(a)
    tr = ["--algo", "ars", "--iters", str(a.iters), "--dirs", str(a.dirs), "--top", str(a.dirs // 2),
          "--workers", str(a.workers)]
    r.log(f"=== run_all: conditions {a.conditions}, {a.seeds} seed(s), {a.iters} iterations, {a.workers} workers")

    for cond in a.conditions:
        for sd in range(a.seeds):
            tag = f"{cond}_s{sd}"
            run = {t: f"runs/{t}_ars_{cond}_synergy_s{sd}" for t in ("reach", "handle", "door")}
            pool_r, pool_h = f"starts/after_reach_{tag}.pkl", f"starts/after_handle_{tag}.pkl"
            base = ["--condition", cond, "--seed", str(sd)]
            r.step(f"{tag}: train reach", ["train", "--task", "reach"] + base + tr + ["--bc-episodes", str(a.bc)],
                   run["reach"] + "/best.npz")
            r.step(f"{tag}: pool after reach", ["starts", "--task", "reach", "--policy", run["reach"],
                                                "--condition", cond, "--episodes", str(a.pool_episodes),
                                                "--out", pool_r], pool_r)
            r.step(f"{tag}: train handle", ["train", "--task", "handle"] + base + tr +
                   ["--bc-episodes", str(a.bc), "--starts", pool_r], run["handle"] + "/best.npz")
            r.step(f"{tag}: pool after handle", ["starts", "--task", "handle", "--policy", run["handle"],
                                                 "--condition", cond, "--episodes", str(a.pool_episodes),
                                                 "--out", pool_h], pool_h)
            r.step(f"{tag}: train door", ["train", "--task", "door"] + base + tr +
                   ["--bc-episodes", str(a.bc), "--starts", pool_h], run["door"] + "/best.npz")
            r.step(f"{tag}: chain (all test doors)",
                   ["chain", "--reach", run["reach"], "--handle", run["handle"], "--door", run["door"],
                    "--doors", "all", "--episodes", str(a.eval_episodes), "--out", f"runs/chain_ars_{tag}"],
                   f"runs/chain_ars_{tag}/chain_eval.csv")

    if not a.no_baselines:
        for sd in range(a.seeds):
            base = ["--condition", "random", "--seed", str(sd)]
            pool_r = f"starts/after_reach_random_s{sd}.pkl"
            # Q2a: the DOF question needs learning FROM SCRATCH (no cloning) for both action spaces,
            # everything else identical (same starts, same budget)
            for mode in ("synergy", "full"):
                r.step(f"Q2a s{sd}: handle from scratch, {mode} actions",
                       ["train", "--task", "handle", "--action-mode", mode, "--tag", "scratch"] + base + tr +
                       ["--starts", pool_r], f"runs/handle_ars_random_{mode}_scratch_s{sd}/best.npz")
            # Q2b: one policy for everything, cloned from the same controller as the chained three
            r.step(f"Q2b s{sd}: one policy for everything",
                   ["train", "--task", "full"] + base + tr + ["--bc-episodes", str(a.bc)],
                   f"runs/full_ars_random_synergy_s{sd}/best.npz")

    # evaluate every trained run on every test-door set, then report
    runs_dir = os.path.join(HERE, "runs")
    if os.path.isdir(runs_dir) or a.dry_run:
        names = sorted(d for d in (os.listdir(runs_dir) if os.path.isdir(runs_dir) else [])
                       if not d.startswith("chain") and os.path.exists(os.path.join(runs_dir, d, "config.json")))
        for d in names:
            r.step(f"evaluate {d}", ["eval", "--run", f"runs/{d}", "--doors", "all",
                                     "--episodes", str(a.eval_episodes)], f"runs/{d}/eval_unseen.csv")
    r.step("report", ["report"], None)
    r.log(f"=== finished in {(time.time() - r.t0) / 3600:.2f} h: {r.done} done, {r.skipped} skipped, "
          f"{len(r.failed)} failed {r.failed if r.failed else ''}")
    r.log("results: results/results.md (tables), results/*.png (plots)")


if __name__ == "__main__":
    main()
