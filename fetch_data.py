#!/usr/bin/env python3
"""The Desk - data pipeline.

Pulls data for five panels, writes small JSON files into data/ (or --out DIR).
Each panel is fetched independently: if one source fails, the previous JSON for
that panel is kept, the error is recorded in log.json, and the other panels
still update.

Usage:
    python fetch_data.py                 # live run (needs FRED_API_KEY, BOK_API_KEY)
    python fetch_data.py --demo --out X  # synthetic data, for previewing index.html offline
"""
import argparse
import json
import os
import sys
import time
import zlib
from datetime import date, datetime, timezone
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).parent
KEEP_POINTS = 520          # ~2y of daily points kept per series in the JSON
UA = "Mozilla/5.0 (compatible; the-desk-dashboard/1.0)"

# ---- Configuration you may want to edit -----------------------------------
FRONTIER_TICKER = "emb.us"   # Stooq symbol for the frontier/EM bond proxy
BOE_SERIES = {                # Bank of England IADB codes (verify at build time)
    "bank_rate": "IUDBEDR",   # Official Bank Rate, daily
    "gilt_5y": "IUDSNPY",     # Nominal par yield, 5y
    "gilt_10y": "IUDMNPY",    # Nominal par yield, 10y
    "gilt_20y": "IUDLNPY",    # Nominal par yield, 20y
}
# ONS Public Sector Finances CDIDs. Fill in / verify the None entries by opening
# the series on ons.gov.uk and copying its 4-letter CDID. None => skipped.
ONS_SERIES = {
    "psnb_ex_banks": "J5II",          # PSNB ex public sector banks, GBP m, monthly
    "central_govt_interest": None,    # TODO: central government debt interest payable
    "net_social_benefits": None,      # TODO: net social benefits paid by central govt
}
ONS_TS_URL = ("https://www.ons.gov.uk/economy/governmentpublicsectorandtaxes/"
              "publicsectorfinance/timeseries/{cdid}/pusf/data")
# Which snapshot series a position can be marked against
LIVE_SERIES = {"usdkrw": "USD/KRW", "ust2": "UST 2Y", "ust10": "UST 10Y",
               "gold": "Gold", "emb": "Frontier proxy", "gilt10": "Gilt 10Y"}
DEFAULT_POSITION_SERIES = {"usdkrw_ndf": "usdkrw"}
# ---------------------------------------------------------------------------

TODAY = datetime.now(timezone.utc).date()
DEMO = False


# ------------------------------- HTTP helpers -------------------------------
def http_get(url, params=None, headers=None, retries=3, timeout=45):
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout,
                             headers={"User-Agent": UA, **(headers or {})})
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"GET {url.split('?')[0]} failed: {last}")


def stooq(symbol):
    """Daily close series from Stooq CSV. Raises if Stooq returns a non-CSV page
    (Stooq sometimes demands an API key; set STOOQ_API_KEY if so)."""
    if DEMO:
        return demo_series(symbol)
    params = {"s": symbol, "i": "d"}
    if os.getenv("STOOQ_API_KEY"):
        params["apikey"] = os.environ["STOOQ_API_KEY"]
    headers = {"Referer": "https://stooq.com/", "Accept": "text/csv,text/plain,*/*"}
    text = None
    last_err = None
    for host in ("stooq.com", "stooq.pl"):  # .com occasionally serves an
        try:                                # interstitial page; .pl mirrors the same data
            text = http_get(f"https://{host}/q/d/l/", params=params, headers=headers).text
            if text.lstrip().startswith("Date,"):
                break
            last_err = f"unexpected response: {text[:200]!r}"
            text = None
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
    if text is None:
        raise RuntimeError(f"Stooq {symbol}: {last_err}")
    df = pd.read_csv(StringIO(text), parse_dates=["Date"]).dropna(subset=["Close"])
    return df.set_index("Date")["Close"].astype(float).sort_index()


def fred(series_id, start="2023-01-01"):
    """FRED series. Uses the official API if FRED_API_KEY is set; otherwise scrapes the
    public fredgraph.csv endpoint (no key needed)."""
    if DEMO:
        return demo_series("fred:" + series_id)
    key = os.getenv("FRED_API_KEY")
    if key:
        j = http_get("https://api.stlouisfed.org/fred/series/observations", params={
            "series_id": series_id, "api_key": key, "file_type": "json",
            "observation_start": start}).json()
        rows = [(o["date"], float(o["value"])) for o in j["observations"] if o["value"] not in (".", "")]
        return pd.Series([v for _, v in rows], index=pd.to_datetime([d for d, _ in rows]), name=series_id)
    text = http_get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                    params={"id": series_id, "cosd": start},
                    headers={"Accept": "text/csv,text/plain,*/*"}, timeout=60).text
    df = pd.read_csv(StringIO(text))
    if df.shape[1] < 2:
        raise RuntimeError(f"FRED csv {series_id}: unexpected response: {text[:80]!r}")
    dates, vals = pd.to_datetime(df.iloc[:, 0]), pd.to_numeric(df.iloc[:, 1], errors="coerce")
    return pd.Series(vals.values, index=dates, name=series_id).dropna()


def boe(codes, start="2023-01-01"):
    """Bank of England IADB CSV export -> DataFrame indexed by date, columns=codes."""
    if DEMO:
        return pd.DataFrame({c: demo_series("boe:" + c) for c in codes})
    d0 = pd.Timestamp(start).strftime("%d/%b/%Y")
    params = {"csv.x": "yes", "Datefrom": d0, "Dateto": "now",
              "SeriesCodes": ",".join(codes), "CSVF": "TN", "UsingCodes": "Y",
              "VPD": "Y", "VFD": "N"}
    text = http_get("https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp",
                    params=params).text
    df = pd.read_csv(StringIO(text))
    df.columns = [c.strip() for c in df.columns]
    if "DATE" not in df.columns:
        raise RuntimeError(f"BoE: unexpected response: {text[:80]!r}")
    df["DATE"] = pd.to_datetime(df["DATE"], format="%d %b %Y")
    return df.set_index("DATE").apply(pd.to_numeric, errors="coerce").sort_index()


def bok_base_rate(start="2023-01-01"):
    """BoK base rate from ECOS (stat 722Y001, item 0101000, daily)."""
    if DEMO:
        return demo_series("bok")
    key = os.getenv("BOK_API_KEY")
    if not key:
        raise RuntimeError("BOK_API_KEY not set")
    d0 = pd.Timestamp(start).strftime("%Y%m%d")
    d1 = TODAY.strftime("%Y%m%d")
    url = f"https://ecos.bok.or.kr/api/StatisticSearch/{key}/json/en/1/3000/722Y001/D/{d0}/{d1}/0101000"
    j = http_get(url).json()
    if "StatisticSearch" not in j:
        raise RuntimeError(f"ECOS: {json.dumps(j)[:120]}")
    rows = j["StatisticSearch"]["row"]
    return pd.Series([float(r["DATA_VALUE"]) for r in rows],
                     index=pd.to_datetime([r["TIME"] for r in rows], format="%Y%m%d")).sort_index()


def ons(cdid):
    if DEMO:
        idx = pd.date_range(end=TODAY, periods=24, freq="MS")
        return pd.Series(np.random.default_rng(zlib.crc32(cdid.encode())).normal(15000, 3000, len(idx)), index=idx)
    j = http_get(ONS_TS_URL.format(cdid=cdid)).json()
    rows = [(pd.to_datetime(m["date"], format="%Y %b"), float(m["value"]))
            for m in j["months"] if m["value"] not in ("", None)]
    return pd.Series([v for _, v in rows], index=[d for d, _ in rows]).sort_index()


# ------------------------------ demo generator ------------------------------
DEMO_BASE = {"usdkrw": (1440, .004), "xauusd": (3900, .009), "emb.us": (92, .003),
             "fred:DGS2": (3.7, .01), "fred:DGS10": (4.2, .008), "fred:DFF": (4.1, 0),
             "fred:CPIAUCSL": (320, .0003), "fred:DFII10": (1.8, .012), "bok": (2.5, 0),
             "boe:IUDBEDR": (3.75, 0), "boe:IUDSNPY": (3.9, .01), "boe:IUDMNPY": (4.5, .008),
             "boe:IUDLNPY": (5.1, .006)}


def demo_series(name):
    base, vol = DEMO_BASE.get(name, (100, .01))
    idx = pd.bdate_range(end=TODAY, periods=700)
    n = len(idx)
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    s = pd.Series(base * np.exp(np.cumsum(rng.normal(0, vol, n)) * (1 if vol else 0)), index=idx)
    if name in ("fred:DFF", "bok", "boe:IUDBEDR"):
        s = pd.Series(np.where(np.arange(n) < n - 90, base + .25, base), index=idx)
    if name == "fred:CPIAUCSL":
        s = s.resample("MS").last().dropna()
    return s


# ------------------------------- shared utils -------------------------------
def pairs(s, n=KEEP_POINTS):
    s = s.dropna().tail(n)
    return [[d.strftime("%Y-%m-%d"), round(float(v), 4)] for d, v in s.items()]


def last(s):
    s = s.dropna()
    return float(s.iloc[-1]), s.index[-1].strftime("%Y-%m-%d")


def load_meetings():
    p = ROOT / "meetings.json"
    return json.loads(p.read_text()) if p.exists() else {}


def rate_status(s, key, bucket=None):
    """Days since last change + next scheduled meeting for a policy rate."""
    s = s.dropna()
    r = (s / bucket).round() * bucket if bucket else s.round(4)
    chg = r[r.diff().fillna(0) != 0]
    out = {"rate": round(float(s.iloc[-1]), 3), "as_of": s.index[-1].strftime("%Y-%m-%d")}
    if len(chg):
        d = chg.index[-1]
        out.update(last_change=d.strftime("%Y-%m-%d"), days_since_change=(TODAY - d.date()).days,
                   last_change_bp=round(float(chg.iloc[-1] - r.shift(1).loc[d]) * 100))
    else:
        out.update(last_change=None, days_since_change=None, last_change_bp=None,
                   note=f"no change in fetched window (since {s.index[0].date()})")
    upcoming = sorted(d for d in load_meetings().get(key, []) if d >= TODAY.isoformat())
    out["next_meeting"] = upcoming[0] if upcoming else None
    out["days_to_meeting"] = (date.fromisoformat(upcoming[0]) - TODAY).days if upcoming else None
    return out


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:  # noqa: BLE001
        return default


class Panel:
    """Collects data, warnings, and the snapshot values used for talking points."""
    def __init__(self):
        self.data, self.warnings, self.snap = {}, [], {}

    def optional(self, label, fn):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            self.warnings.append(f"{label}: {e}")
            return None



def _numbers(cells):
    out = []
    for c in cells:
        t = str(c).replace(",", "").replace("£", "").strip()
        try:
            out.append(float(t))
        except ValueError:
            pass
    return out


def scrape_dmo():
    """Best-effort scrape of the DMO: total gilts in issue (report D1A export) and the
    current-year gross financing remit (currentremit.pdf). Every figure is sanity-checked;
    anything that fails is left None so data/dmo_manual.json values are used instead."""
    if DEMO:
        return {"gilts_in_issue_gbp_bn": 2500.0, "remit_gbp_bn": 300.0, "source": "demo"}
    out, notes = {"source": "DMO (scraped)"}, []
    try:
        r = http_get("https://www.dmo.gov.uk/data/ExportReport", params={"reportCode": "D1A"})
        try:
            tables = pd.read_html(StringIO(r.text))
        except Exception:  # noqa: BLE001 - not HTML, try Excel
            tables = list(pd.read_excel(BytesIO(r.content), sheet_name=None, header=None).values())
        best = None
        for t in tables:
            for row in t.astype(str).values.tolist():
                if str(row[0]).strip().lower().startswith("total") and _numbers(row[1:]):
                    best = max(_numbers(row[1:]))      # nominal amount outstanding, GBP m
        if best and 1_500 <= best / 1000 <= 3_500:
            out["gilts_in_issue_gbp_bn"] = round(best / 1000, 1)
        else:
            notes.append(f"gilts in issue: no plausible Total row found (got {best})")
    except Exception as e:  # noqa: BLE001
        notes.append(f"gilts in issue: {e}")
    try:
        import re
        import pdfplumber
        pdf = http_get("https://www.dmo.gov.uk/dmo_static_reports/currentremit.pdf").content
        with pdfplumber.open(BytesIO(pdf)) as f:
            text = "\n".join((pg.extract_text() or "") for pg in f.pages[:5])
        m = re.search(r"(20\d\d-\d\d)", text)
        # DMO remit wording varies by year ("gross financing requirement of £X billion",
        # "gilt sales in 2025-26 are planned to be £X billion", "gross gilt issuance of £Xbn", ...).
        # Try progressively looser patterns until one lands in the plausible range.
        patterns = [
            r"gross\s+(?:financing\s+requirement|gilt\s+(?:sales|issuance))[^\n]{0,80}?"
            r"(?:£|GBP)\s?([\d,]+\.?\d*)\s?(?:billion|bn)\b",
            r"(?:£|GBP)\s?([\d,]+\.?\d*)\s?(?:billion|bn)[^\n]{0,60}?gross\s+gilt",
            r"(?:gross|total)[^\n]{0,60}?(?:£|GBP)?\s?([\d,]+\.?\d*)\s?(?:billion|bn|m)\b",
        ]
        val = None
        for pat in patterns:
            g = re.search(pat, text, re.I)
            if g:
                val = float(g.group(1).replace(",", ""))
                break
        if val and val > 5_000:
            val /= 1000                                # was in GBP m
        if val and 100 <= val <= 500:
            out["remit_gbp_bn"], out["remit_fiscal_year"] = round(val, 1), (m.group(1) if m else None)
        else:
            snippet = re.sub(r"\s+", " ", text)[:200]
            notes.append(f"remit: no plausible gross figure found (got {val}); text starts {snippet!r}")
    except Exception as e:  # noqa: BLE001
        notes.append(f"remit: {e}")
    out["scrape_notes"] = notes
    return out


def dmo_data(p):
    """Scraped DMO values, falling back per-field to the hand-edited manual file."""
    manual = read_json(ROOT / "data" / "dmo_manual.json", {}) or {}
    got = p.optional("DMO scrape", scrape_dmo) or {}
    for n in got.get("scrape_notes", []):
        p.warnings.append("DMO " + n)
    merged = {**manual}
    for k in ("gilts_in_issue_gbp_bn", "remit_gbp_bn", "remit_fiscal_year"):
        if got.get(k) is not None:
            merged[k] = got[k]
    merged["scraped_fields"] = [k for k in ("gilts_in_issue_gbp_bn", "remit_gbp_bn") if got.get(k) is not None]
    return merged


# --------------------------------- panels -----------------------------------
def panel_korea():
    p = Panel()
    fx = stooq("usdkrw")
    v, d = last(fx)
    p.snap["usdkrw"] = v
    p.data = {"latest": {"usdkrw": v, "as_of": d},
              "series": {"usdkrw": pairs(fx)}}
    bok = p.optional("BoK base rate", bok_base_rate)
    if bok is not None and len(bok):
        p.data["series"]["bok_rate"] = pairs(bok)
        p.data["bok"] = rate_status(bok, "bok")
    return p


def panel_us():
    p = Panel()
    d2, d10, dff = fred("DGS2"), fred("DGS10"), fred("DFF")
    p.snap.update(ust2=last(d2)[0], ust10=last(d10)[0])
    both = pd.concat([d2, d10], axis=1, keys=["2y", "10y"]).ffill().dropna()
    curve = (both["10y"] - both["2y"]) * 100
    p.snap["curve_2s10s"] = float(curve.iloc[-1])
    p.data = {"latest": {"ust2": last(d2), "ust10": last(d10),
                         "curve_2s10s_bp": round(float(curve.iloc[-1]), 1)},
              "series": {"ust2": pairs(d2), "ust10": pairs(d10), "curve_2s10s_bp": pairs(curve)},
              "fed": rate_status(dff, "fed", bucket=0.25)}
    p.data["series"]["dff"] = pairs(dff)
    cpi = p.optional("CPI", lambda: fred("CPIAUCSL", start="2020-01-01"))
    if cpi is not None:
        yoy = (cpi.pct_change(12) * 100).dropna()
        p.data["series"]["cpi_yoy"] = pairs(yoy, 60)
        p.data["latest"]["cpi_yoy"] = [round(float(yoy.iloc[-1]), 2), yoy.index[-1].strftime("%Y-%m-%d")]
    return p


def panel_gold():
    p = Panel()
    gold = stooq("xauusd")
    real = fred("DFII10")
    p.snap.update(gold=last(gold)[0], real10=last(real)[0])
    p.data = {"latest": {"gold": last(gold), "real10": last(real)},
              "series": {"gold": pairs(gold), "real10": pairs(real)}}
    wgc = read_json(ROOT / "data" / "wgc_manual.json")
    if wgc is None:
        p.warnings.append("wgc_manual.json missing")
    p.data["wgc"] = wgc
    return p


def panel_frontier():
    p = Panel()
    etf = stooq(FRONTIER_TICKER)
    ust = fred("DGS10")
    p.snap["emb"] = last(etf)[0]
    df = pd.concat([etf, ust], axis=1, keys=["etf", "ust"]).ffill().dropna()
    ret = pd.DataFrame({"etf": df["etf"].pct_change(), "ust": df["ust"].diff()}).dropna()
    corr = ret["etf"].rolling(60).corr(ret["ust"]).dropna()
    if len(corr):
        p.snap["corr60"] = float(corr.iloc[-1])
    p.data = {"ticker": FRONTIER_TICKER.upper(),
              "corr_method": "rolling 60-day correlation of ETF daily % return vs daily change in DGS10 (pp)",
              "latest": {"etf": last(etf), "corr60": round(float(corr.iloc[-1]), 3) if len(corr) else None},
              "series": {"etf": pairs(etf), "corr60": pairs(corr)}}
    return p


def panel_uk():
    p = Panel()
    df = boe(list(BOE_SERIES.values()))
    inv = {v: k for k, v in BOE_SERIES.items()}
    df = df.rename(columns=inv)
    br = df["bank_rate"].dropna()
    g10 = df["gilt_10y"].dropna()
    p.snap["gilt10"] = float(g10.iloc[-1])
    p.snap["bank_rate"] = float(br.iloc[-1])
    curve = {}
    for label, offset in (("now", 0), ("1m ago", 21), ("1y ago", 252)):
        row = df[["gilt_5y", "gilt_10y", "gilt_20y"]].dropna()
        if len(row) > offset:
            r = row.iloc[-1 - offset]
            curve[label] = {"date": row.index[-1 - offset].strftime("%Y-%m-%d"),
                            "5y": round(r["gilt_5y"], 3), "10y": round(r["gilt_10y"], 3),
                            "20y": round(r["gilt_20y"], 3)}
    p.data = {"latest": {"gilt10": last(g10), "bank_rate": last(br)},
              "series": {"gilt10": pairs(g10), "bank_rate": pairs(br)},
              "curve": curve, "boe": rate_status(br, "boe"), "ons": {}, "dmo": dmo_data(p)}
    for k, cdid in ONS_SERIES.items():
        if not cdid:
            p.warnings.append(f"ONS {k}: CDID not configured")
            continue
        s = p.optional(f"ONS {k}", lambda c=cdid: ons(c))
        if s is not None and len(s):
            v, d = last(s)
            p.data["ons"][k] = {"cdid": cdid, "latest": v, "as_of": d, "series": pairs(s, 36)}
    return p


PANELS = [("korea_fx", panel_korea), ("us_rates", panel_us), ("gold_real_yields", panel_gold),
          ("frontier", panel_frontier), ("uk_gilts", panel_uk)]


# ------------------------- talking points / positions -----------------------
def pct(a, b):
    return (a / b - 1) * 100


def talking_points(snap, prev, positions, status):
    """One plain-English sentence per panel: this run vs the previous run."""
    tp = {}
    if not prev:
        return {k: "First run - baseline recorded; deltas start next update." for k in
                ("korea_fx", "us_rates", "gold_real_yields", "frontier", "uk_gilts")}

    def has(*ks):
        return all(k in snap and k in prev for k in ks)

    def dirn(x, up="up", down="down"):
        return up if x > 0 else down if x < 0 else "unchanged"

    if has("usdkrw"):
        ch = pct(snap["usdkrw"], prev["usdkrw"])
        s = f"USD/KRW moved {ch:+.1f}% since the last update to {snap['usdkrw']:,.0f}"
        for k, pos in positions.items():
            if pos.get("series", DEFAULT_POSITION_SERIES.get(k)) == "usdkrw" and pos.get("target"):
                toward = (pos["target"] - snap["usdkrw"]) * (pos["target"] - prev["usdkrw"]) >= 0 and \
                    abs(pos["target"] - snap["usdkrw"]) < abs(pos["target"] - prev["usdkrw"])
                s += f", {'toward' if toward else 'away from'} the {pos['target']:,.0f} target"
                break
        tp["korea_fx"] = s + "."
    if has("ust10"):
        bp = (snap["ust10"] - prev["ust10"]) * 100
        s = f"The US 10Y yield is {dirn(bp)} {abs(bp):.0f}bp to {snap['ust10']:.2f}%"
        if "curve_2s10s" in snap:
            s += f"; 2s10s sits at {snap['curve_2s10s']:+.0f}bp"
        fed = status.get("fed")
        if fed and fed.get("days_since_change") is not None:
            s += f"; Fed funds unchanged for {fed['days_since_change']} days" if fed["days_since_change"] else ""
        tp["us_rates"] = s + "."
    if has("gold", "real10"):
        ch = pct(snap["gold"], prev["gold"])
        bp = (snap["real10"] - prev["real10"]) * 100
        tp["gold_real_yields"] = (f"Gold {dirn(ch, 'rose', 'fell')} {abs(ch):.1f}% to ${snap['gold']:,.0f} "
                                  f"while the 10Y real yield {dirn(bp, 'rose', 'fell')} {abs(bp):.0f}bp to {snap['real10']:.2f}%.")
    if has("emb"):
        ch = pct(snap["emb"], prev["emb"])
        s = f"The frontier proxy {dirn(ch, 'rose', 'fell')} {abs(ch):.1f}% since the last update"
        if "corr60" in snap:
            s += f"; 60-day correlation with UST 10Y changes is {snap['corr60']:+.2f}"
        tp["frontier"] = s + "."
    if has("gilt10"):
        bp = (snap["gilt10"] - prev["gilt10"]) * 100
        s = f"The 10Y gilt yield is {dirn(bp)} {abs(bp):.0f}bp to {snap['gilt10']:.2f}%"
        boe_s = status.get("boe")
        if boe_s and boe_s.get("days_since_change") is not None:
            s += f"; Bank Rate {boe_s['rate']:.2f}%, {boe_s['days_since_change']} days since the last move"
        tp["uk_gilts"] = s + "."
    return tp


def position_pnl(positions, snap):
    out = {}
    for k, pos in positions.items():
        entry = {"detail": {kk: vv for kk, vv in pos.items()}}
        sk = pos.get("series") or DEFAULT_POSITION_SERIES.get(k)
        if sk in snap and pos.get("entry_level") and pos.get("direction") in ("long", "short"):
            cur, e = snap[sk], float(pos["entry_level"])
            sign = 1 if pos["direction"] == "long" else -1
            pnl_pct = sign * (cur / e - 1) * 100
            entry.update(series=sk, current=cur, pnl_pct=round(pnl_pct, 2))
            notional = float(pos.get("notional", 1_000_000))
            # USD/KRW quoted KRW per USD: P&L in USD on a USD notional = notional*(cur-e)/cur for a long-USD position
            if sk == "usdkrw":
                entry["pnl_usd"] = round(sign * notional * (cur - e) / cur, 0)
                entry["notional"] = notional
            if pos.get("target"):
                t = float(pos["target"])
                if t != e:
                    entry["progress_to_target_pct"] = round((cur - e) / (t - e) * 100, 1)
        else:
            entry["note"] = "no live comparable series - tracked as a view"
        out[k] = entry
    return out


def update_streak(log):
    """Streak = consecutive business days with a fully successful update."""
    prev_d = log.get("last_success_date")
    today = TODAY.isoformat()
    if prev_d == today:
        return log.get("streak", 1)
    if prev_d and len(pd.bdate_range(prev_d, today)) == 2:
        return log.get("streak", 0) + 1
    return 1


# ----------------------------------- main -----------------------------------
def main():
    global DEMO
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="synthetic data (offline preview only)")
    ap.add_argument("--out", default=str(ROOT / "data"))
    a = ap.parse_args()
    DEMO = a.demo
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if DEMO and out.resolve() == (ROOT / "data").resolve():
        sys.exit("Refusing to write demo data into data/. Use --out.")

    log = read_json(out / "log.json", {})
    positions = read_json(ROOT / "positions.json", {})
    now = datetime.now(timezone.utc)
    snap, status, panel_log, all_ok = {}, {}, {}, True

    for name, fn in PANELS:
        try:
            p = fn()
            payload = {"panel": name, "updated": now.isoformat(timespec="seconds"),
                       "demo": DEMO, "warnings": p.warnings, **p.data}
            (out / f"{name}.json").write_text(json.dumps(payload, separators=(",", ":")))
            snap.update(p.snap)
            for k in ("fed", "bok", "boe"):
                if k in p.data:
                    status[k] = p.data[k]
            panel_log[name] = {"ok": True, "updated": payload["updated"], "warnings": p.warnings}
            print(f"[ok]   {name}" + (f"  warnings: {p.warnings}" if p.warnings else ""))
        except Exception as e:  # noqa: BLE001
            all_ok = False
            old = log.get("panels", {}).get(name, {})
            panel_log[name] = {"ok": False, "error": str(e), "updated": old.get("updated")}
            print(f"[FAIL] {name}: {e}", file=sys.stderr)

    snaps = log.get("snapshots", {})
    today = TODAY.isoformat()
    prior_dates = sorted(d for d in snaps if d < today)
    prev = snaps[prior_dates[-1]] if prior_dates else None
    tp = talking_points(snap, prev, positions, status)
    snaps[today] = {**snaps.get(today, {}), **snap}
    for d in sorted(snaps)[:-30]:
        del snaps[d]

    new = {"last_run": now.isoformat(timespec="seconds"), "demo": DEMO, "all_ok": all_ok,
           "panels": panel_log, "talking_points": tp,
           "compared_with": prior_dates[-1] if prior_dates else None,
           "positions": position_pnl(positions, {**(prev or {}), **snap}),
           "snapshots": snaps,
           "streak": log.get("streak", 0), "last_success_date": log.get("last_success_date"),
           "best_streak": log.get("best_streak", 0)}
    if all_ok:
        new["streak"] = update_streak(log)
        new["last_success_date"] = today
    elif new["last_success_date"] and len(pd.bdate_range(new["last_success_date"], today)) > 2:
        new["streak"] = 0    # a missed business day breaks the streak
    new["best_streak"] = max(new["best_streak"], new["streak"])
    (out / "log.json").write_text(json.dumps(new, indent=1))
    print(f"streak={new['streak']} all_ok={all_ok}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
