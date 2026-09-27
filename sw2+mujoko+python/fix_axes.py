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
import math
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


def perpendicular_axes(rec, parent, child, ref=None, prefer=None):
    """Universal joint in CAD = point Coincident + Perpendicular between an axis fixed in the
    parent and an axis fixed in the child (and NO axis-to-axis Coincident). Returns
    (bend, sideways, how) or None. The BEND axis is the one closest to `ref` (the next joint's
    hinge, e.g. the thumb IP - it bends in the same plane); `prefer` = "parent"/"child" forces it."""
    last = lambda s: str(s or "").split("/")[-1]
    mates = (rec or {}).get("mates") or []
    if any(m.get("type") == "COINCIDENT" and sum(et in (1, 2, 4) for et in m.get("etypes", [])) >= 2
           for m in mates):
        return None                                   # a real axis-to-axis hinge: not a universal joint
    for m in mates:
        if m.get("type") != "PERPENDICULAR":
            continue
        owners = list(m.get("owners") or []) + [""] * 4
        ents = [(owners[i], unit(d)) for i, (et, d) in enumerate(zip(m.get("etypes", []), m.get("dirs", [])))
                if et in (1, 2, 4)]
        if len(ents) != 2 or ents[0][1] is None or ents[1][1] is None:
            continue
        own = [last(o) for o, _ in ents]
        k_par = own.index(last(parent.name)) if last(parent.name) in own else \
            (1 - own.index(last(child.name)) if last(child.name) in own else None)
        if prefer in ("parent", "child") and k_par is not None:
            k = k_par if prefer == "parent" else 1 - k_par
            how = f"{prefer}-side axis (--uj-bend {prefer})"
        elif ref is not None:
            k = max((0, 1), key=lambda i: abs(float(ents[i][1] @ ref)))
            side = "" if k_par is None else (" = parent-side" if k == k_par else " = child-side")
            how = f"axis closest to the next joint's hinge{side}"
        elif k_par is not None:
            k, how = k_par, "parent-side axis"
        else:
            continue
        return ents[k][1], ents[1 - k][1], f"Perpendicular mate (universal joint): {how}"
    return None

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pkg_dir")
    ap.add_argument("--config", help="joints yaml (default: <pkg_dir>/*.joints.yaml)")
    ap.add_argument("--write", action="store_true", help="write proposed axes into the yaml")
    ap.add_argument("--ref-axis", action="append", default=[], metavar="CHILD=AXIS",
                    help="take a joint's axis from a reference axis drawn in the TOP assembly (Hand_forearm). "
                         "CHILD = part of the joint's child link name, AXIS = the reference axis name, e.g. "
                         "Thumb_Proximal_1=thumb_mcp_flex. Repeatable.")
    ap.add_argument("--uj-bend", choices=["auto", "parent", "child"], default="auto",
                    help="universal joints: which Perpendicular-mate axis is the BEND axis (default auto = "
                         "the one closest to the next joint's hinge)")
    ap.add_argument("--from-mates", action="append", default=[], metavar="CHILD",
                    help="always take this joint's axis + pivot from its own coincident mates (point + axis), even if "
                         "sw2robot found an axis. CHILD = part of the joint's child link name, e.g. handle_1. Repeatable.")
    ap.add_argument("--list-axes", action="store_true", help="list the top-assembly reference axes in the extract")
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

    # universal joints (point + Perpendicular, no axis-to-axis Coincident): always take the axis from
    # the mate, even if sw2robot's classifier returned some axis of its own for the pair
    for d in info:
        if d["rec"] is None or d["entry"].get("type", "fixed") == "fixed":
            continue
        if perpendicular_axes(d["rec"], d["parent"], d["child"], ref=np.array([1.0, 0.0, 0.0])) is None:
            continue
        if d["status"] in ("sw2robot finds it", None):
            d["sw_axis"] = d["axis"]
            d["axis"], d["status"], d["uj"] = None, None, True

    # --from-mates: force the axis from the joint's own coincident point + axis mates
    for key in args.from_mates:
        hits = [d for d in info if d["child"] is not None and d["entry"].get("type") != "fixed"
                and key.lower() in (d["child"].link_name + " " + d["child"].name).lower()]
        if len(hits) != 1:
            sys.exit(f"--from-mates: '{key}' matches {len(hits)} joints (child link names)")
        d = hits[0]
        if d["rec"] is None:
            sys.exit(f"--from-mates: no mates between {d['entry'].get('parent')} and {d['entry'].get('child')}")
        d["sw_axis"] = d["axis"]
        d["axis"], d["status"], d["force_mates"] = None, None, True

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
            if d.get("force_mates"):
                coinc = [e for e in ents if e[0] == "COINCIDENT"]
                cd = line_dirs(coinc)
                if len(cd) == 1:
                    prop[id(d)] = (point_on_axis(coinc, cd[0]), cd[0], "--from-mates: the joint's own coincident axis")
                else:
                    prop[id(d)] = (centre, None, f"--from-mates: {len(cd)} coincident axes found - can't choose")
                continue
            dirs = line_dirs(ents)
            uj = perpendicular_axes(d["rec"], d["parent"], d["child"], nxt[0] if nxt else None,
                                    None if args.uj_bend == "auto" else args.uj_bend)

            if uj is not None:
                d["sideways"] = uj[1]
                prop[id(d)] = (centre, uj[0], f"{uj[2]} = BEND axis")
            elif len(dirs) == 1:
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

    # ---- joints whose axis is taken from a CAD reference axis (top assembly)
    ref_axes = {str(a.name): a for a in (getattr(graph, "reference_axes", None) or [])}
    if args.list_axes:
        print("\ntop-assembly reference axes in this extract:")
        for nm, a in ref_axes.items():
            print(f"   {nm:<30} point {mm(a.document_point)}  dir {np.round(unit(a.document_direction), 3).tolist()}")
        if not ref_axes:
            print("   (none - create them in Hand_forearm.SLDASM itself, then re-extract)")
    forced = []
    for spec in args.ref_axis:
        key, _, axname = spec.partition("=")
        hits = [d for d in info if d["child"] is not None and d["entry"].get("type") != "fixed"
                and key.lower() in (d["child"].link_name + " " + d["child"].name).lower()]
        if len(hits) != 1:
            sys.exit(f"--ref-axis: '{key}' matches {len(hits)} joints (child link names) - be more specific")
        if axname not in ref_axes:
            sys.exit(f"--ref-axis: no top-assembly reference axis named '{axname}'. "
                     f"Available: {sorted(ref_axes) or 'none'}")
        a = ref_axes[axname]
        d = hits[0]
        d["axis"], d["status"] = None, None
        prop[id(d)] = (np.asarray(a.document_point, float), unit(a.document_direction),
                       f"CAD reference axis '{axname}'")
        forced.append(d)
        print(f"[ref-axis] {d['entry'].get('parent')} -> {d['entry'].get('child')}  <-  '{axname}'")

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
            nx = [n for n in by_parent.get(d["child"].name, []) if axis_of(n) is not None]
            if d.get("uj") and ax is not None and nx:
                ipa = axis_of(nx[0])
                a1 = math.degrees(math.acos(min(1.0, abs(float(ax @ ipa)))))
                a2 = (math.degrees(math.acos(min(1.0, abs(float(d["sideways"] @ ipa)))))
                      if d.get("sideways") is not None else float("nan"))
                print(f"         vs next joint's hinge: bend axis {a1:.1f} deg, sideways axis {a2:.1f} deg "
                      f"(bend should be the small one)")
            if d.get("sw_axis") is not None and ax is not None:
                print(f"         (sw2robot had picked {np.round(d['sw_axis'], 3).tolist()}, "
                      f"{math.degrees(math.acos(min(1.0, abs(float(d['sw_axis'] @ ax))))):.1f} deg off - replaced)")
            if d.get("sideways") is not None:
                ang_ = math.degrees(math.acos(min(1.0, abs(float(d["sideways"] @ ax))))) if ax is not None else 0
                print(f"         sideways axis: {np.round(d['sideways'], 4).tolist()}  "
                      f"[{ang_:.1f} deg to the bend axis] -> added in MuJoCo with --add-axis")
        print()

    if args.write:
        n = 0
        extra = [f for f in forced if f not in todo] + [x for x in info if x.get("force_mates") and x not in todo]
        for d in todo + extra:
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
