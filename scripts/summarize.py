"""Collect exploration results (explore_*.json) into a markdown table and a CSV.

    python scripts/summarize.py runs/explore --out runs/explore/summary
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

COLUMNS = [
    ("size", "orchard"),
    ("seed", "seed"),
    ("start", "start"),
    ("sensing", "depth / pose"),
    ("variant", "variant"),
    ("status", "status"),
    ("rows", "rows est / true"),
    ("length", "length err"),
    ("width", "width err"),
    ("row_spacing", "row spacing err"),
    ("tree_spacing", "tree spacing err"),
    ("distance", "flown (m)"),
    ("coverage", "coverage"),
    ("collisions", "collisions"),
    ("drift", "VO drift"),
    ("trunks", "trunk P / R (ICP)"),
    ("trunk_err", "trunk err (m)"),
    ("fps", "agent Hz"),
]

COMPACT = [
    ("orchard", "orchard (true size)"),
    ("start", "start"),
    ("sensing_label", "sensing"),
    ("rows", "rows est / true"),
    ("length", "length err"),
    ("width", "width err"),
    ("row_spacing", "row spacing err"),
    ("distance", "flown (m)"),
    ("coverage", "coverage"),
    ("collisions", "collisions"),
    ("drift", "VO drift"),
    ("trunks", "trunk P / R"),
]


def row_of(path: Path) -> dict:
    d = json.loads(path.read_text())
    name = path.stem.replace("explore_", "").split("_")
    s = d.get("size") or {}
    icp = d.get("trunks_icp_aligned") or {}
    ms = sum(d.get(k, 0.0) for k in ("ms_depth", "ms_vo", "ms_map"))
    pct = lambda k: f"{s[k]:+.1f} %" if k in s else "-"  # noqa: E731
    gt = d.get("orchard", {})
    return {
        "orchard": f"{name[0]} #{name[1].lstrip('s')}: {gt.get('n_rows')} rows, "
                   f"{gt.get('length_m', 0):.0f} x {gt.get('width_m', 0):.0f} m",
        "sensing_label": ("camera only" if d.get("depth_source") == "net"
                          else "perfect depth + pose"),
        "size": name[0],
        "seed": name[1].lstrip("s"),
        "start": d.get("start", name[2]),
        "sensing": f"{d.get('depth_source')} / {d.get('pose_source')}",
        "variant": "_".join(name[5:]) or "-",
        "status": d.get("status"),
        "rows": f"{s.get('n_rows_est', '-')} / {s.get('n_rows_true', '-')}",
        "length": pct("length_m_err_pct"),
        "width": pct("width_m_err_pct"),
        "row_spacing": pct("row_spacing_m_err_pct"),
        "tree_spacing": pct("tree_spacing_m_err_pct"),
        "distance": f"{d.get('distance_m', 0):.0f}",
        "coverage": f"{100 * d.get('coverage', 0):.0f} %",
        "collisions": str(d.get("collisions")),
        "drift": f"{d.get('vo_drift_pct', float('nan')):.1f} %",
        "trunks": f"{icp.get('precision', 0):.2f} / {icp.get('recall', 0):.2f}",
        "trunk_err": f"{icp.get('mean_error_m', float('nan')):.2f}",
        "fps": f"{1000.0 / ms:.1f}" if ms else "-",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("runs", nargs="+", help="directories with explore_*.json")
    ap.add_argument("--out", default="summary")
    ap.add_argument("--compact", action="store_true", help="fewer columns, for the README")
    args = ap.parse_args()
    files = sorted(f for d in args.runs for f in Path(d).glob("explore_*.json"))
    order = {"small": 0, "medium": 1, "large": 2}
    def key(r: dict) -> tuple:  # camera-only runs first, then by size and seed
        return r["sensing"] != "net / vo", order.get(r["size"], 9), int(r["seed"])

    rows = sorted((row_of(f) for f in files), key=key)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)
    cols = COMPACT if args.compact else COLUMNS
    lines = ["| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(r[k] for k, _ in cols) + " |" for r in rows]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
