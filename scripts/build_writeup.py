"""Fill the one-page write-up with results and print it to PDF (headless Edge or Chrome).

    python scripts/build_writeup.py --summary runs/final/summary.csv \
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


def table_html(rows: list[dict]) -> str:
    head = ("<tr><th>orchard (true size)</th><th>sensing</th><th class=n>rows</th>"
            "<th class=n>length</th><th class=n>width</th><th class=n>flown</th>"
            "<th class=n>coverage</th><th class=n>coll.</th></tr>")
    body = []
    for r in rows:
        sensing = "camera only" if r["sensing"].startswith("net") else "perfect depth+pose"
        body.append(
            f"<tr><td>{r['size']} #{r['seed']} ({r['start']})</td><td>{sensing}</td>"
            f"<td class=n>{r['rows']}</td><td class=n>{r['length']}</td>"
            f"<td class=n>{r['width']}</td><td class=n>{r['distance']} m</td>"
            f"<td class=n>{r['coverage']}</td><td class=n>{r['collisions']}</td></tr>")
    return f"<table>{head}{''.join(body)}</table>"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--summary", required=True, help="CSV from scripts/summarize.py")
    ap.add_argument("--figure", default=None)
    ap.add_argument("--caption", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--template", default="docs/writeup/writeup.html")
    ap.add_argument("--out", default="docs/OrchardScout_writeup.pdf")
    args = ap.parse_args()

    with open(args.summary, newline="") as fh:
        rows = list(csv.DictReader(fh))
    html = Path(args.template).read_text(encoding="utf-8")
    html = html.replace("<!--RESULTS_TABLE-->", table_html(rows)
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
