#!/usr/bin/env python3
"""
fix_axes.py
===========
For every revolute joint in the joints yaml that sw2robot could not give an axis
("no CAD axis, defaulting to world +Z"), this script reads the mate geometry from
graph.json (after sub-assembly expansion, in world coordinates), shows why the
axis failed, and proposes axis_point / axis_dir:

  * from a reference axis / edge used in a coincident mate      -> "mate axis"
  * several non-parallel axes mated (hinge locked in CAD)        -> the one parallel to
                                                                    the neighbouring joint
  * point-only joint, next joint in the chain has an axis        -> parallel to it
  * point-only joint, previous joint has an axis (the wrist)     -> perpendicular to it

Proposals feed each other over several passes (MCP axis -> PIP -> DIP).

Run it once to read the report; add --write to put the proposals into the yaml
(original saved as <yaml>.bak). Then rebuild and check every slider.

Usage:
    python fix_axes.py <pkg_dir>              # report only
    python fix_axes.py <pkg_dir> --write      # also write axis_point/axis_dir into the yaml
    python fix_axes.py <pkg_dir> --reset --write   # after a re-extract: drop old overrides, recompute
"""
import argparse
import glob
import os
import shutil
import sys

import numpy as np
import yaml

ETYPE = {0: "point", 1: "line", 2: "circle", 3: "plane", 4: "cylinder", 5: "sphere", 7: "cone", None: "?"}


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else None


def mm(p):
    return "[" + ", ".join(f"{x * 1000:7.2f}" for x in p) + "] mm"


def entities(rec):
    """All mate entities of an edge as (mate_type, etype, point, dir)."""
    out = []
    for m in (rec or {}).get("mates") or []:
        for et, p, d in zip(m.get("etypes", []), m.get("points", []), m.get("dirs", [])):
            out.append((m.get("type"), et, np.asarray(p, float), unit(d)))
    return out


def line_dirs(ents):
    """Distinct (non-parallel) directions of line-like mate entities."""
    out = []
    for _, et, _, d in ents:
        if et in (1, 2, 4) and d is not None and all(abs(float(d @ e)) < 0.99 for e in out):
            out.append(d)
    return out


def point_on_axis(ents, d):
    """A point on the joint axis: the mated point if there is one, else a line's point."""
    pts = [p for _, et, p, _ in ents if et == 0]
    lps = [p for _, et, p, dd in ents if et in (1, 2, 4) and dd is not None and abs(float(dd @ d)) > 0.99]
    if pts:
        c = np.mean(pts, axis=0)
        if lps:                                   # project the point onto the mated line
            c = lps[0] + d * float((c - lps[0]) @ d)
        return c
    return lps[0] if lps else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pkg_dir")
    ap.add_argument("--config", help="joints yaml (default: <pkg_dir>/*.joints.yaml)")
    ap.add_argument("--write", action="store_true", help="write proposed axes into the yaml")
    ap.add_argument("--reset", action="store_true",
                    help="ignore (and with --write, remove) every axis_point/axis_dir already in the yaml. "
                         "Use after a re-extract: overrides are world coordinates and go stale if the CAD moved")
    args = ap.parse_args()

    try:
        from sw2robot.exporter.model import classify_edge_geo, from_graph
        from sw2robot.exporter.state import GraphState
    except ImportError:
        sys.exit("needs sw2robot:  python -m pip install sw2robot")

    pkg = os.path.abspath(args.pkg_dir)
    cfg_path = args.config or (sorted(glob.glob(os.path.join(pkg, "*.joints.yaml"))) or [None])[0]
    if not cfg_path:
        sys.exit("no *.joints.yaml in the package folder")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if args.reset:
        for j in cfg.get("joints") or []:
            j.pop("axis_point", None)
            j.pop("axis_dir", None)
        print("[reset] ignoring all axis_point/axis_dir overrides in the yaml")

    graph = GraphState.load(os.path.join(pkg, "graph.json"))
    comps, adjacency, _ = from_graph(graph, expand=cfg.get("expand"), no_expand=cfg.get("no_expand"))
    alias = {}
    for c in comps:
        alias[c.name] = c
        alias[c.link_name] = c

    # ---- what sw2robot itself resolves
    info = []
    for j in cfg.get("joints") or []:
        pa, ch = alias.get(str(j.get("parent"))), alias.get(str(j.get("child")))
        rec = adjacency.get(frozenset((pa.name, ch.name))) if pa and ch else None
        d = dict(entry=j, parent=pa, child=ch, rec=rec, axis=None, status=None, note=None)
        if j.get("type", "fixed") == "fixed":
            d["status"] = "fixed"
        elif j.get("axis_dir"):
            d["axis"] = unit(j["axis_dir"])
            d["status"] = "set in yaml"
        elif pa is None or ch is None:
            d["status"] = "UNKNOWN parent/child name (check spelling against the editor tree)"
        elif rec is None:
            d["status"] = "no mates between these parts"
        else:
            geo = None
            try:
                geo = classify_edge_geo(rec.get("mates") or [])
            except Exception:
                pass
            if geo and geo[1] is not None:
                d["axis"] = unit(geo[1][1])
                d["status"] = "sw2robot finds it"
            else:
                d["note"] = geo[2] if geo else "no usable mate geometry"
        info.append(d)

    by_child = {d["child"].name: d for d in info if d["child"]}
    by_parent = {}
    for d in info:
        if d["parent"]:
            by_parent.setdefault(d["parent"].name, []).append(d)

    # ---- proposals, iterated so they can feed their neighbours
    prop = {}                                    # id(d) -> (point, dir or None, reason)

    def axis_of(d):
        if d is None:
            return None
        if d["axis"] is not None:
            return d["axis"]
        p = prop.get(id(d))
        return p[1] if p else None

    todo = [d for d in info if d["axis"] is None and d["status"] is None]
    for _ in range(5):
        for d in todo:
            if prop.get(id(d)) and prop[id(d)][1] is not None:
                continue
            ents = entities(d["rec"])
            pts = [p for _, et, p, _ in ents if et == 0]
            centre = np.mean(pts, axis=0) if pts else d["child"].world[:3, 3]
            nxt = [a for a in (axis_of(n) for n in by_parent.get(d["child"].name, [])) if a is not None]
            prv = by_child.get(d["parent"].name)
            prv_ax = axis_of(prv)
            dirs = line_dirs(ents)

            if len(dirs) == 1:
                prop[id(d)] = (point_on_axis(ents, dirs[0]), dirs[0], "from the mated reference axis")
            elif len(dirs) > 1:
                ref = nxt[0] if nxt else prv_ax
                if ref is not None:
                    best = max(dirs, key=lambda v: abs(float(v @ ref)))
                    prop[id(d)] = (point_on_axis(ents, best), best,
                                   f"{len(dirs)} non-parallel axes are mated (hinge locked in CAD); "
                                   "picked the one parallel to the neighbouring joint - CHECK")
                else:
                    prop[id(d)] = (centre, None, f"{len(dirs)} non-parallel axes are mated - "
                                   "hinge is locked in CAD; remove one of those coincident mates")
            elif nxt:
                prop[id(d)] = (centre, nxt[0], "guess: parallel to the next joint in the chain")
            elif prv is not None and prv_ax is not None and prv["parent"] is not None:
                bone = unit(centre - prv["parent"].world[:3, 3])
                ax = unit(np.cross(prv_ax, bone)) if bone is not None else None
                if ax is not None:
                    prop[id(d)] = (centre, ax, "guess: perpendicular to the previous joint (2-DOF wrist)")
            if id(d) not in prop and pts:
                prop[id(d)] = (centre, None, "only a point is mated - direction unknown")

    # ---- report
    print(f"\nconfig: {cfg_path}\n")
    for d in info:
        j = d["entry"]
        name = f"{j.get('parent')}  ->  {j.get('child')}"
        if d["status"] == "fixed":
            continue
        if d["axis"] is not None:
            print(f"OK    {name}\n      axis ({d['status']}): {np.round(d['axis'], 3).tolist()}\n")
            continue
        print(f"FIX   {name}")
        if d["status"]:
            print(f"      {d['status']}")
        if d["note"]:
            print(f"      sw2robot: {d['note']}")
        seen = set()
        for mt, et, p, dd in entities(d["rec"]):
            key = (mt, et, tuple(np.round(p, 5)), None if dd is None else tuple(np.round(np.abs(dd), 3)))
            if key in seen:
                continue
            seen.add(key)
            ds = "" if dd is None else f"  dir {np.round(dd, 3).tolist()}"
            print(f"      mate {mt:<12} {ETYPE.get(et, et):<8} at {mm(p)}{ds}")
        p = prop.get(id(d))
        if p:
            pt, ax, why = p
            print(f"      -> {why}")
            if pt is not None:
                print(f"         axis_point: {np.round(pt, 6).tolist()}")
            if ax is not None:
                print(f"         axis_dir:   {np.round(ax, 6).tolist()}")
        print()

    if args.write:
        n = 0
        for d in todo:
            p = prop.get(id(d))
            if p and p[0] is not None and p[1] is not None:
                d["entry"]["axis_point"] = [round(float(x), 6) for x in p[0]]
                d["entry"]["axis_dir"] = [round(float(x), 6) for x in p[1]]
                n += 1
        shutil.copy2(cfg_path, cfg_path + ".bak")
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=None)
        print(f"wrote {n} axis override(s) into {cfg_path}  (backup: {cfg_path}.bak)")
    else:
        print("report only - rerun with --write to put the proposals into the yaml")


if __name__ == "__main__":
    main()
