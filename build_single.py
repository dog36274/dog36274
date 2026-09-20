#!/usr/bin/env python3
"""Bundle index.html + Chart.js + data/*.json into ONE self-contained the-desk.html
that works offline by double-click (file://), no server needed.

    python fetch_data.py && python build_single.py            # -> the-desk.html
    python build_single.py --data /tmp/demo --out demo.html    # from another data dir
"""
import argparse, json, re
from pathlib import Path

ROOT = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(ROOT / "data"))
ap.add_argument("--out", default=str(ROOT / "the-desk.html"))
a = ap.parse_args()

names = ["korea_fx", "us_rates", "gold_real_yields", "frontier", "uk_gilts", "log"]
data = {}
for n in names:
    p = Path(a.data) / f"{n}.json"
    data[n] = json.loads(p.read_text()) if p.exists() else None

html = (ROOT / "index.html").read_text()
chartjs = (ROOT / "vendor" / "chart.umd.js").read_text().replace("</script>", "<\\/script>")
payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
html, n = re.subn(r'<!--CHARTJS--><script src="[^"]+"></script>',
                  lambda m: f"<script>{chartjs}</script>\n<script>window.__DESK_DATA__={payload};</script>", html)
assert n == 1, "Chart.js placeholder not found in index.html"
Path(a.out).write_text(html)
print(f"wrote {a.out} ({len(html)/1024:.0f} KB)")
