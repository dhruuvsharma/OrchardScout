"""Fill the one-page write-up with results and print it to PDF (headless Edge or Chrome).

    python scripts/build_writeup.py --summary dev=runs/final/summary.csv \
        --summary held-out=runs/heldout/summary.csv \
        --figure docs/images/explore_medium_map.png --out docs/OrchardScout_writeup.pdf
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "msedge", "chromium", "google-chrome",
]


def _rows_cell(r: dict | None) -> str:
    if r is None:
        return "-"
    est, true = (x.strip() for x in r["rows"].split("/"))
    cls = "" if est == true else ' class="bad"'
    return f"<span{cls}>{est}</span>"


def table_html(sets: list[tuple[str, list[dict]]]) -> str:
    """One row per orchard: camera-only result next to the perfect-sensing bound."""
    head = (
        "<tr><th>set</th><th>orchard (true size)</th><th>start</th>"
        "<th class=n>rows<br>camera</th><th class=n>length<br>camera</th>"
        "<th class=n>width<br>camera</th><th class=n>flown<br>camera</th>"
        "<th class=n>coverage<br>camera</th><th class=n>collisions<br>camera</th>"
        "<th class=n>rows<br>perfect</th><th class=n>length<br>perfect</th>"
        "<th class=n>width<br>perfect</th></tr>")
    body = []
    for name, rows in sets:
        by = {}
        for r in rows:
            by.setdefault((r["size"], r["seed"]), {})[r["sensing"]] = r
        first = True
        for (_size, _seed), pair in by.items():
            cam, gt = pair.get("net / vo"), pair.get("gt / gt")
            ref = cam or gt
            sep = ' class="sep"' if first else ""
            first = False
            c = (lambda k, r=cam: r[k] if r else "-")  # noqa: E731
            g = (lambda k, r=gt: r[k] if r else "-")  # noqa: E731
            body.append(
                f"<tr{sep}><td>{name}</td><td>{ref['orchard']}</td><td>{ref['start']}</td>"
                f"<td class=n>{_rows_cell(cam)} / {ref['rows'].split('/')[1].strip()}</td>"
                f"<td class=n>{c('length')}</td><td class=n>{c('width')}</td>"
                f"<td class=n>{c('distance')} m</td><td class=n>{c('coverage')}</td>"
                f"<td class=n>{c('collisions')}</td>"
                f"<td class=n>{_rows_cell(gt)}</td><td class=n>{g('length')}</td>"
                f"<td class=n>{g('width')}</td></tr>")
    return f"<table>{head}{''.join(body)}</table>"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--summary", action="append", required=True,
                    help="NAME=summary.csv from scripts/summarize.py (repeatable)")
    ap.add_argument("--figure", default=None)
    ap.add_argument("--caption", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--template", default="docs/writeup/writeup.html")
    ap.add_argument("--out", default="docs/OrchardScout_writeup.pdf")
    args = ap.parse_args()

    sets = []
    for item in args.summary:
        name, path = item.split("=", 1)
        with open(path, newline="") as fh:
            sets.append((name, list(csv.DictReader(fh))))
    html = Path(args.template).read_text(encoding="utf-8")
    html = html.replace("<!--RESULTS_TABLE-->", table_html(sets)
                        + (f'<p class="note">{args.note}</p>' if args.note else ""))
    if args.figure:
        fig = Path(args.figure).resolve().as_uri()
        html = html.replace("<!--RESULTS_FIGURE-->",
                            f'<figure><img src="{fig}"><figcaption>{args.caption}'
                            "</figcaption></figure>")
    filled = Path(args.template).with_name("writeup_filled.html")
    filled.write_text(html, encoding="utf-8")

    browser = next((b for b in BROWSERS if Path(b).exists() or shutil.which(b)), None)
    if browser is None:
        raise SystemExit(f"no Edge/Chrome found; open {filled} and print to PDF")
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([browser, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={out}", filled.resolve().as_uri()], check=True,
                   timeout=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
