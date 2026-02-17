import os
import json
import time
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import requests

LAT = float(os.getenv("LAT", "40.78858"))
LON = float(os.getenv("LON", "-73.9661"))
NY = ZoneInfo("America/New_York")

DB_PATH = os.getenv("DB_PATH", "/var/data/weatheredge.sqlite")
TOMORROW_KEY = os.getenv("TOMORROW_KEY", "")

# --------- HTTP safe get ---------

def safe_get(url: str, headers=None, timeout=15, retries=3, backoff=1.8):
    headers = headers or {}
    for i in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)

            if r.status_code == 429 or (500 <= r.status_code <= 599):
                time.sleep(backoff * (i + 1))
                continue

            return r
        except requests.RequestException:
            time.sleep(backoff * (i + 1))
    return None

# --------- DB ---------

def db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS hourly_forecast(
            source TEXT,
            time_utc TEXT,
            temp_f REAL,
            fetched_at_utc TEXT
        )
    """)
    con.commit()
    return con

def insert_rows(con, source, rows):
    if not rows:
        return 0
    fetched = datetime.now(timezone.utc).isoformat()
    con.executemany(
        "INSERT INTO hourly_forecast(source,time_utc,temp_f,fetched_at_utc) VALUES (?,?,?,?)",
        [(source, t, tf, fetched) for (t, tf) in rows]
    )
    con.commit()
    return len(rows)

def load_recent(con, hours=6):
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    cur = con.execute(
        "SELECT source, time_utc, temp_f FROM hourly_forecast WHERE fetched_at_utc >= ?",
        (cutoff,)
    )
    by_source = {}
    for src, t, tf in cur.fetchall():
        by_source.setdefault(src, []).append((datetime.fromisoformat(t), float(tf)))
    for src in by_source:
        by_source[src].sort(key=lambda x: x[0])
    return by_source

# --------- time helpers ---------

def in_horizon(dt: datetime, horizon_hours=48):
    now = datetime.now(timezone.utc)
    end = now + timedelta(hours=horizon_hours)
    return now <= dt <= end

def next_n_hours(series, n=10):
    now = datetime.now(timezone.utc)
    out = [(dt, tf) for dt, tf in series if dt >= now]
    return out[:n]

def day_minmax_ny(series, day_offset=1):
    """
    day_offset=1 => tomorrow in NY
    Returns (min,max) in Fahrenheit or (None,None)
    """
    target = (datetime.now(NY) + timedelta(days=day_offset)).date()
    vals = []
    for dt_utc, tf in series:
        dt_ny = dt_utc.astimezone(NY)
        if dt_ny.date() == target:
            vals.append(float(tf))
    if not vals:
        return None, None
    return float(min(vals)), float(max(vals))

# --------- fetchers ---------

def fetch_open_meteo(horizon_hours=48):
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        "&hourly=temperature_2m"
        "&temperature_unit=fahrenheit"
        "&timezone=UTC"
    )
    r = safe_get(url)
    if not r:
        return []

    try:
        j = r.json()
        rows = []
        for t, tf in zip(j["hourly"]["time"], j["hourly"]["temperature_2m"]):
            dt = datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
            if in_horizon(dt, horizon_hours):
                rows.append((dt.isoformat(), float(tf)))
        return rows
    except Exception:
        return []

def fetch_weather_gov(horizon_hours=48):
    headers = {"User-Agent": "WeatherEdge", "Accept": "application/geo+json"}
    p = safe_get(f"https://api.weather.gov/points/{LAT},{LON}", headers=headers)
    if not p:
        return []

    try:
        forecast_url = p.json()["properties"]["forecastHourly"]
    except Exception:
        return []

    r = safe_get(forecast_url, headers=headers)
    if not r:
        return []

    try:
        periods = r.json()["properties"]["periods"]
        rows = []
        for it in periods:
            dt = datetime.fromisoformat(it["startTime"]).astimezone(timezone.utc)
            if in_horizon(dt, horizon_hours):
                rows.append((dt.isoformat(), float(it["temperature"])))
        return rows
    except Exception:
        return []

def fetch_tomorrow_io(horizon_hours=48):
    if not TOMORROW_KEY:
        print("Tomorrow.io: Missing API key")
        return []

    url = "https://api.tomorrow.io/v4/timelines"

    payload = {
        "location": f"{LAT},{LON}",
        "fields": ["temperature"],
        "timesteps": ["1h"],
        "units": "imperial"
    }

    headers = {
        "Content-Type": "application/json",
        "apikey": TOMORROW_KEY
    }

    r = safe_get(url, headers=headers)

    if not r:
        print("Tomorrow.io request failed")
        return []

    if r.status_code != 200:
        print("Tomorrow.io ERROR:", r.status_code, r.text)
        return []

    try:
        j = r.json()

        rows = []
        for it in j["data"]["timelines"][0]["intervals"]:
            dt = datetime.fromisoformat(it["startTime"].replace("Z", "+00:00"))
            tf = float(it["values"]["temperature"])

            if in_horizon(dt, horizon_hours):
                rows.append((dt.isoformat(), tf))

        print("Tomorrow.io OK rows:", len(rows))
        return rows

    except Exception as e:
        print("Tomorrow.io parse error:", e)
        return []

# --------- stats & betting ---------

def remove_outliers_iqr(values_dict):
    vals = np.array(list(values_dict.values()), dtype=float)
    if len(vals) < 3:
        return values_dict
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    filtered = {k: v for k, v in values_dict.items() if lo <= v <= hi}
    return filtered if len(filtered) >= 2 else values_dict

def monte_carlo_prob_over(threshold_f, mean_f, sigma_f, sims=30000, seed=7):
    rng = np.random.default_rng(seed)
    draws = rng.normal(loc=mean_f, scale=max(0.1, sigma_f), size=sims)
    return float(np.mean(draws > threshold_f))

def read_market_price():
    try:
        with open("/app/market.json", "r", encoding="utf-8") as f:
            j = json.load(f)
        return float(j["market_price_yes_over_48"])
    except Exception:
        return float(os.getenv("MARKET_PRICE_YES_OVER_48", "0.17"))

def fractional_kelly(p, price, fraction=0.25):
    if not (0 < price < 1):
        return 0.0
    raw = (p - price) / (1 - price)
    return max(0.0, raw) * fraction

# --------- result model ---------

@dataclass
class SourceSummary:
    src: str
    rows_inserted: int
    next_hours: list  # list[(dt_iso, temp)]
    tmr_min: float | None
    tmr_max: float | None

@dataclass
class RunResult:
    ok: bool
    generated_at_utc: str
    lat: float
    lon: float
    ny_date_tomorrow: str
    sources: list  # list[SourceSummary]
    consensus_min: float | None
    consensus_max: float | None
    removed_outliers: list
    threshold_f: float
    p_over: float | None
    market_price: float | None
    edge: float | None
    stake: float | None
    notes: list

# --------- main runner ---------

def run_once_struct() -> RunResult:
    con = db()
    notes = []

    inserts = {
        "open_meteo": insert_rows(con, "open_meteo", fetch_open_meteo()),
        "weather_gov": insert_rows(con, "weather_gov", fetch_weather_gov()),
        "tomorrow_io": insert_rows(con, "tomorrow_io", fetch_tomorrow_io()),
    }

    by_source = load_recent(con, hours=6)

    src_summaries = []
    per_min = {}
    per_max = {}

    for src, series in by_source.items():
        nxt = [(dt.isoformat(), float(tf)) for dt, tf in next_n_hours(series, 10)]
        tmin, tmax = day_minmax_ny(series, day_offset=1)

        src_summaries.append(
            SourceSummary(
                src=src,
                rows_inserted=inserts.get(src, 0),
                next_hours=nxt,
                tmr_min=tmin,
                tmr_max=tmax
            )
        )

        if tmin is not None and tmax is not None:
            per_min[src] = tmin
            per_max[src] = tmax

    # consensus min/max (outlier filter on both)
    consensus_min = consensus_max = None
    removed = []

    if per_min and per_max:
        min_filtered = remove_outliers_iqr(per_min)
        max_filtered = remove_outliers_iqr(per_max)

        removed = sorted(set(per_min.keys()) - set(min_filtered.keys()) |
                         set(per_max.keys()) - set(max_filtered.keys()))

        consensus_min = float(np.mean(list(min_filtered.values())))
        consensus_max = float(np.mean(list(max_filtered.values())))
    else:
        notes.append("Not enough tomorrow (NY) data to compute MIN/MAX consensus.")

    # probability model (use consensus_max as tmax mean)
    threshold = 48.0
    p_over = market = edge = stake = None

    if consensus_max is not None:
        # spread from sources (std of max), plus base sigma
        vals = list(per_max.values())
        spread = float(np.std(vals)) if len(vals) > 1 else 2.0
        base_sigma = 2.0
        sigma_total = float(np.sqrt(base_sigma**2 + spread**2))

        p_over = monte_carlo_prob_over(threshold, consensus_max, sigma_total)
        market = read_market_price()
        edge = p_over - market
        stake = fractional_kelly(p_over, market, fraction=0.25)
    else:
        notes.append("Skipped probability model because consensus_max is None.")

    tomorrow_ny = (datetime.now(NY) + timedelta(days=1)).date().isoformat()

    return RunResult(
        ok=True,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        lat=LAT,
        lon=LON,
        ny_date_tomorrow=tomorrow_ny,
        sources=src_summaries,
        consensus_min=consensus_min,
        consensus_max=consensus_max,
        removed_outliers=removed,
        threshold_f=threshold,
        p_over=p_over,
        market_price=market,
        edge=edge,
        stake=stake,
        notes=notes
    )

def run_once_text() -> str:
    # Keep a compact text output for /api/run
    rr = run_once_struct()
    lines = []
    lines.append(f"Generated: {rr.generated_at_utc}")
    lines.append(f"Location: {rr.lat},{rr.lon}")
    lines.append(f"Tomorrow (NY): {rr.ny_date_tomorrow}")

    for s in rr.sources:
        lines.append(f"\n[{s.src}] inserted={s.rows_inserted}")
        if s.tmr_min is not None and s.tmr_max is not None:
            lines.append(f"Tomorrow MIN/MAX: {s.tmr_min:.1f}F / {s.tmr_max:.1f}F")
        else:
            lines.append("Tomorrow MIN/MAX: N/A")

    if rr.consensus_min is not None and rr.consensus_max is not None:
        lines.append(f"\nConsensus MIN/MAX: {rr.consensus_min:.1f}F / {rr.consensus_max:.1f}F")
    else:
        lines.append("\nConsensus MIN/MAX: N/A")

    if rr.p_over is not None:
        lines.append(f"P(TMAX > {rr.threshold_f:.0f}F): {rr.p_over*100:.2f}%")
        lines.append(f"Market YES: {rr.market_price*100:.2f}% | Edge: {rr.edge*100:.2f}%")
        lines.append(f"Stake (25% Kelly): {rr.stake*100:.2f}%")
    else:
        lines.append("Model probability: N/A")

    if rr.removed_outliers:
        lines.append(f"Outliers removed: {rr.removed_outliers}")

    if rr.notes:
        lines.append("\nNotes:")
        lines.extend([f"- {x}" for x in rr.notes])

    return "\n".join(lines)

if __name__ == "__main__":
    print(run_once_text())
