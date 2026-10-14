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

Order per condition (fixed / random) and seed, in the planned hand's action space (design):
  reach -> start pool after reach -> handle (from that pool) -> pool after handle -> door (from it) -> chain
  (every stage learns from scratch by default; --bc-stages reach door starts those two from cloning)
  python run_all.py --runs-dir runs_v4 --no-baselines   # a new run beside the old one, Q1 only
  (v4 default: the skills ON - a real grasp on the handle, the hand not the forearm on the door - and all three
   stages cloned from the hand-written controller, then fine-tuned)
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
        """Run `experiment.py args` unless `output` exists. A training step counts as done only when its DONE
        marker exists (written after the LAST iteration) - a run cut off half-way is trained again from the start.
        Runs from before DONE existed count as done if their progress.csv reached the final iteration."""
        if output and output.endswith("/DONE") and not os.path.exists(os.path.join(HERE, output)):
            prog = os.path.join(HERE, os.path.dirname(output), "progress.csv")
            want = args[args.index("--iters") + 1] if "--iters" in args else None
            if want and os.path.exists(prog):
                rows = [l.split(",")[0] for l in open(prog).read().splitlines()[1:] if l.strip()]
                if rows and rows[-1] == want:
                    open(os.path.join(HERE, output), "w").write("finished (older run, from progress.csv)\n")
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
    ap.add_argument("--mode", choices=["design", "synergy", "full"], default="design",
                    help="action space of the pipeline (default design = the planned hand)")
    ap.add_argument("--iters", type=int, default=150, help="ARS iterations for reach / door (default 150)")
    ap.add_argument("--handle-iters", type=int, default=300,
                    help="ARS iterations for handle, full, and door when it learns from scratch (default 300)")
    ap.add_argument("--bc-stages", nargs="*", default=["reach", "handle", "door"],
                    help="stages started from behaviour cloning, then fine-tuned by ARS (default: all three - with the "
                         "v4 grasp rule a wrapping grasp is hard to find by chance; the hand-written 'wrap' controller "
                         "demonstrates it). '--bc-stages' with nothing after it = everything from scratch (v3)")
    ap.add_argument("--no-skills", action="store_true",
                    help="v1-v3 rules: no grasp needed on the handle, the forearm may push the door")
    ap.add_argument("--runs-dir", default="runs", help="where runs and start pools go (e.g. runs_v3)")
    ap.add_argument("--results-dir", default=None, help="report folder (default results, or results_<runs-dir>)")
    ap.add_argument("--train-eval-episodes", type=int, default=20, help="evaluation episodes during training")
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
    tr = ["--algo", "ars", "--dirs", str(a.dirs), "--top", str(a.dirs // 2), "--workers", str(a.workers),
          "--eval-episodes", str(a.train_eval_episodes)] + (["--no-grasp", "--no-palm-push", "--no-door-wrist"]
                                                            if a.no_skills else [])
    R = a.runs_dir.rstrip("/\\")
    iters = lambda task: str(a.handle_iters if task == "handle" or (task == "door" and task not in a.bc_stages)
                             else a.iters)
    bc = lambda task: (["--bc-episodes", str(a.bc)] if task in a.bc_stages else ["--tag", "scratch"])
    tag = lambda task: "" if task in a.bc_stages else "_scratch"
    M = a.mode
    r.log(f"=== run_all: mode {M}, conditions {a.conditions}, {a.seeds} seed(s), {a.iters} iterations "
          f"(handle {a.handle_iters}), cloning for {a.bc_stages}, skills {'OFF' if a.no_skills else 'ON'}, "
          f"{a.workers} workers")

    for cond in a.conditions:
        for sd in range(a.seeds):
            tg = f"{cond}_s{sd}"
            run = {t: f"{R}/{t}_ars_{cond}_{M}{tag(t)}_s{sd}" for t in ("reach", "handle", "door")}
            pool_r, pool_h = f"{R}/starts/after_reach_{M}_{tg}.pkl", f"{R}/starts/after_handle_{M}_{tg}.pkl"
            base = ["--condition", cond, "--seed", str(sd), "--action-mode", M, "--runs", R]
            r.step(f"{tg}: train reach", ["train", "--task", "reach", "--iters", iters("reach")] + base + tr + bc("reach"),
                   run["reach"] + "/DONE")
            r.step(f"{tg}: pool after reach", ["starts", "--task", "reach", "--policy", run["reach"], "--action-mode", M,
                                               "--condition", cond, "--episodes", str(a.pool_episodes),
                                               "--out", pool_r], pool_r)
            r.step(f"{tg}: train handle", ["train", "--task", "handle", "--iters", iters("handle")] + base + tr +
                   bc("handle") + ["--starts", pool_r], run["handle"] + "/DONE")
            r.step(f"{tg}: pool after handle", ["starts", "--task", "handle", "--policy", run["handle"],
                                                "--action-mode", M, "--condition", cond,
                                                "--episodes", str(a.pool_episodes), "--out", pool_h], pool_h)
            r.step(f"{tg}: train door", ["train", "--task", "door", "--iters", iters("door")] + base + tr + bc("door") +
                   ["--starts", pool_h], run["door"] + "/DONE")
            r.step(f"{tg}: chain (all test doors)",
                   ["chain", "--reach", run["reach"], "--handle", run["handle"], "--door", run["door"],
                    "--doors", "all", "--episodes", str(a.eval_episodes), "--out", f"{R}/chain_ars_{tg}"],
                   f"{R}/chain_ars_{tg}/chain_eval.csv")

    if not a.no_baselines:
        for sd in range(a.seeds):
            base = ["--condition", "random", "--seed", str(sd), "--runs", R]
            pool_r = f"{R}/starts/after_reach_{M}_random_s{sd}.pkl"
            # Q2a: the handle stage FROM SCRATCH with each action space (same starts, same budget). The main
            # pipeline's random-condition handle run is already the from-scratch run of mode M.
            for mode in ("design", "synergy", "full"):
                if mode == M and "handle" not in a.bc_stages:
                    continue
                r.step(f"Q2a s{sd}: handle from scratch, {mode} actions",
                       ["train", "--task", "handle", "--action-mode", mode, "--tag", "scratch",
                        "--iters", iters("handle")] + base + tr + ["--starts", pool_r],
                       f"{R}/handle_ars_random_{mode}_scratch_s{sd}/DONE")
            # Q2b: one policy for the whole sequence (cloned from the same controller)
            r.step(f"Q2b s{sd}: one policy for everything",
                   ["train", "--task", "full", "--action-mode", M, "--iters", str(a.handle_iters)] + base + tr +
                   ["--bc-episodes", str(a.bc)], f"{R}/full_ars_random_{M}_s{sd}/DONE")

    # evaluate every trained run on every test-door set, then report
    runs_dir = os.path.join(HERE, R)
    if os.path.isdir(runs_dir) or a.dry_run:
        names = sorted(d for d in (os.listdir(runs_dir) if os.path.isdir(runs_dir) else [])
                       if not d.startswith("chain") and os.path.exists(os.path.join(runs_dir, d, "config.json")))
        for d in names:
            r.step(f"evaluate {d}", ["eval", "--run", f"{R}/{d}", "--doors", "all",
                                     "--episodes", str(a.eval_episodes)], f"{R}/{d}/eval_unseen.csv")
    res = a.results_dir or ("results" if R == "runs" else f"results_{os.path.basename(R)}")
    r.step("report", ["report", "--runs", R, "--out", res], None)
    r.log(f"=== finished in {(time.time() - r.t0) / 3600:.2f} h: {r.done} done, {r.skipped} skipped, "
          f"{len(r.failed)} failed {r.failed if r.failed else ''}")
    r.log(f"results: {res}/results.md (tables), {res}/*.png (plots)")


if __name__ == "__main__":
    main()
