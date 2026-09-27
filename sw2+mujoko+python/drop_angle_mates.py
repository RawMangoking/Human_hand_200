#!/usr/bin/env python3
"""
drop_angle_mates.py
===================
sw2robot 0.4.4 reads a SolidWorks LimitAngle mate inside a sub-assembly as a
plain, RIGID angle constraint. On a hinge defined by coincident point + coincident
axis, that extra constraint removes the one free rotation, so the classifier finds
no axis and every joint falls back to "world +Z".

This script deletes the ANGLE mates from the extracted graph.json (a backup is
kept as graph.json.bak) so the build derives each hinge axis from your
coincident-axis mates. Joint limits come from Hand_forearm.joints.yaml instead,
so nothing is lost. Your CAD files are not touched.

Usage (from the package folder):
    python drop_angle_mates.py            # patch ./graph.json
    python drop_angle_mates.py --restore  # put the original back
"""
import argparse
import json
import os
import shutil
import sys


def patch(node, removed):
    """Walk the graph; on every mate edge {a, b, types, mates} drop ANGLE mates."""
    if isinstance(node, dict):
        if {"a", "b", "types"} <= node.keys():
            mates = node.get("mates") or []
            keep = [m for m in mates if not (isinstance(m, dict) and m.get("type") == "ANGLE")]
            n = len(mates) - len(keep)
            if n or "ANGLE" in node["types"]:
                node["mates"] = keep
                node["types"] = [t for t in node["types"] if t != "ANGLE"]
                removed.append((node["a"], node["b"], n))
        for v in node.values():
            patch(v, removed)
    elif isinstance(node, list):
        for v in node:
            patch(v, removed)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pkg_dir", nargs="?", default=".", help="package folder containing graph.json")
    ap.add_argument("--restore", action="store_true", help="restore graph.json from graph.json.bak")
    args = ap.parse_args()

    graph = os.path.join(os.path.abspath(args.pkg_dir), "graph.json")
    backup = graph + ".bak"
    if args.restore:
        if not os.path.exists(backup):
            sys.exit("no graph.json.bak to restore")
        shutil.copy2(backup, graph)
        print("restored graph.json from graph.json.bak")
        return
    if not os.path.exists(graph):
        sys.exit(f"not found: {graph}")
    with open(graph, encoding="utf-8") as f:
        data = json.load(f)
    removed = []
    patch(data, removed)
    if removed:
        # a fresh (unpatched) extraction: refresh the backup so --restore gives THIS extraction back
        shutil.copy2(graph, backup)
        print(f"backup -> {backup}")
        with open(graph, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    if not removed:
        print("no ANGLE mates found (already patched?)")
    for a, b, n in removed:
        print(f"  removed {n} ANGLE mate record(s): {a}  <->  {b}")
    print(f"patched {len(removed)} edge(s) in {graph}")


if __name__ == "__main__":
    main()
