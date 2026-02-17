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

EXPECTED_SOURCES = ["open_meteo", "weather_gov", "tomorrow_io"]

# -------------------- HTTP safe get --------------------

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

def safe_post(url: str, headers=None, json_body=None, timeout=15, retries=3, backoff=1.8):
    headers = headers or {}
    for i in range(retries):
        try:
            r = requests.post(url, headers=headers, json=json_body, timeout=timeout)
            if r.status_code == 429 or (500 <= r.status_code <= 599):
                time.sleep(backoff * (i + 1))
                continue
            return r
        except requests.RequestException:
            time.sleep(backoff * (i + 1))
    return None

# -------------------- DB --------------------

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

# -------------------- time helpers --------------------

def in_horizon(dt: datetime, horizon_hours=48):
    now = datetime.now(timezone.utc)
    end = now + timedelta(hours=horizon_hours)
    return now <= dt <= end

def next_n_hours(series, n=10):
    now = datetime.now(timezone.utc)
    out = [(dt, tf) for dt, tf in series if dt >= now]
    return out[:n]

def day_minmax_ny(series, day_offset=1):
    target = (datetime.now(NY) + timedelta(days=day_offset)).date()
    vals = []
    for dt_utc, tf in series:
        dt_ny = dt_utc.astimezone(NY)
        if dt_ny.date() == target:
            vals.append(float(tf))
    if not vals:
        return None, None
    return float(min(vals)), float(max(vals))

def last_hours_now_minmax(series, hours=8):
    """
    Return (now_temp, min_last_h, max_last_h) in last N hours (UTC window)
    now_temp = latest datapoint <= now, or None if not found
    """
    if not series:
        return None, None, None

    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours)

    # min/max within last hours
    window_vals = [tf for dt, tf in series if start <= dt <= now]
    mn = float(min(window_vals)) if window_vals else None
    mx = float(max(window_vals)) if window_vals else None

    # "now" = latest value <= now
    past = [(dt, tf) for dt, tf in series if dt <= now]
    now_tf = float(past[-1][1]) if past else None

    return now_tf, mn, mx

# -------------------- fetchers --------------------

def fetch_open_meteo(horizon_hours=48):
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        "&hourly=temperature_2m"
        "&temperature_unit=fahrenheit"
        "&timezone=UTC"
    )
    r = safe_get(url)
    if not r or r.status_code != 200:
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
    if not p or p.status_code != 200:
        return []
    try:
        forecast_url = p.json()["properties"]["forecastHourly"]
    except Exception:
        return []
    r = safe_get(forecast_url, headers=headers)
    if not r or r.status_code != 200:
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
    # Use v4/timelines (recommended)
    if not TOMORROW_KEY:
        return []

    url = "https://api.tomorrow.io/v4/timelines"
    payload = {
        "location": f"{LAT},{LON}",
        "fields": ["temperature"],
        "timesteps": ["1h"],
        "units": "imperial",
    }
    headers = {"Content-Type": "application/json", "apikey": TOMORROW_KEY}

    r = safe_post(url, headers=headers, json_body=payload)
    if not r or r.status_code != 200:
        return []

    try:
        j = r.json()
        intervals = j["data"]["timelines"][0]["intervals"]
        rows = []
        for it in intervals:
            dt = datetime.fromisoformat(it["startTime"].replace("Z", "+00:00"))
            tf = float(it["values"]["temperature"])
            if in_horizon(dt, horizon_hours):
                rows.append((dt.isoformat(), tf))
        return rows
    except Exception:
        return []

# -------------------- stats & betting --------------------

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

# -------------------- result model --------------------

@dataclass
class SourceSummary:
    src: str
    rows_inserted: int
    status: str  # OK / EMPTY / NO_KEY
    now_f: float | None
    last8h_min_f: float | None
    last8h_max_f: float | None
    tmr_min_f: float | None
    tmr_max_f: float | None

@dataclass
class RunResult:
    ok: bool
    generated_at_utc: str
    lat: float
    lon: float

    ny_date_tomorrow: str

    sources: list  # list[SourceSummary]
    removed_outliers: list
    notes: list

    # Consensus now & last 8h
    consensus_now_f: float | None
    consensus_last8h_min_f: float | None
    consensus_last8h_max_f: float | None

    # Tomorrow consensus
    consensus_tmr_min_f: float | None
    consensus_tmr_max_f: float | None

    # Betting
    threshold_f: float
    p_over: float | None
    market_price: float | None
    edge: float | None
    stake: float | None

# -------------------- main runner --------------------

def run_once_struct() -> RunResult:
    con = db()
    notes = []

    inserts = {
        "open_meteo": insert_rows(con, "open_meteo", fetch_open_meteo()),
        "weather_gov": insert_rows(con, "weather_gov", fetch_weather_gov()),
        "tomorrow_io": insert_rows(con, "tomorrow_io", fetch_tomorrow_io()),
    }

    by_source = load_recent(con, hours=6)

    # per-source now/8h + tomorrow
    per_now = {}
    per_8h_min = {}
    per_8h_max = {}
    per_tmr_min = {}
    per_tmr_max = {}

    src_summaries = []

    for src in EXPECTED_SOURCES:
        series = by_source.get(src, [])
        status = "OK"
        if src == "tomorrow_io" and not TOMORROW_KEY:
            status = "NO_KEY"
        elif not series:
            status = "EMPTY"

        now_f, mn8, mx8 = last_hours_now_minmax(series, hours=8)
        tmin, tmax = day_minmax_ny(series, day_offset=1)

        if now_f is not None:
            per_now[src] = now_f
        if mn8 is not None:
            per_8h_min[src] = mn8
        if mx8 is not None:
            per_8h_max[src] = mx8
        if tmin is not None:
            per_tmr_min[src] = tmin
        if tmax is not None:
            per_tmr_max[src] = tmax

        src_summaries.append(
            SourceSummary(
                src=src,
                rows_inserted=inserts.get(src, 0),
                status=status,
                now_f=now_f,
                last8h_min_f=mn8,
                last8h_max_f=mx8,
                tmr_min_f=tmin,
                tmr_max_f=tmax,
            )
        )

    removed = []

    # Consensus now + last 8h (simple mean with outlier filter when enough)
    consensus_now = None
    consensus_8h_min = None
    consensus_8h_max = None

    if per_now:
        per_now_f = remove_outliers_iqr(per_now)
        removed = sorted(set(removed) | (set(per_now.keys()) - set(per_now_f.keys())))
        consensus_now = float(np.mean(list(per_now_f.values())))

    if per_8h_min:
        per_min_f = remove_outliers_iqr(per_8h_min)
        removed = sorted(set(removed) | (set(per_8h_min.keys()) - set(per_min_f.keys())))
        consensus_8h_min = float(np.mean(list(per_min_f.values())))

    if per_8h_max:
        per_max_f = remove_outliers_iqr(per_8h_max)
        removed = sorted(set(removed) | (set(per_8h_max.keys()) - set(per_max_f.keys())))
        consensus_8h_max = float(np.mean(list(per_max_f.values())))

    # Tomorrow consensus min/max
    consensus_tmr_min = consensus_tmr_max = None
    if per_tmr_min and per_tmr_max:
        tmr_min_f = remove_outliers_iqr(per_tmr_min)
        tmr_max_f = remove_outliers_iqr(per_tmr_max)
        removed = sorted(set(removed) |
                         (set(per_tmr_min.keys()) - set(tmr_min_f.keys())) |
                         (set(per_tmr_max.keys()) - set(tmr_max_f.keys())))
        consensus_tmr_min = float(np.mean(list(tmr_min_f.values())))
        consensus_tmr_max = float(np.mean(list(tmr_max_f.values())))
    else:
        notes.append("Not enough tomorrow (NY) data to compute consensus MIN/MAX.")

    # Betting model based on consensus tomorrow MAX
    threshold = 48.0
    p_over = market = edge = stake = None

    if consensus_tmr_max is not None:
        vals = list(per_tmr_max.values())
        spread = float(np.std(vals)) if len(vals) > 1 else 2.0
        base_sigma = 2.0
        sigma_total = float(np.sqrt(base_sigma**2 + spread**2))

        p_over = monte_carlo_prob_over(threshold, consensus_tmr_max, sigma_total)
        market = read_market_price()
        edge = p_over - market
        stake = fractional_kelly(p_over, market, fraction=0.25)
    else:
        notes.append("Skipped probability model because consensus_tmr_max is None.")

    tomorrow_ny = (datetime.now(NY) + timedelta(days=1)).date().isoformat()

    return RunResult(
        ok=True,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        lat=LAT,
        lon=LON,
        ny_date_tomorrow=tomorrow_ny,

        sources=src_summaries,
        removed_outliers=removed,
        notes=notes,

        consensus_now_f=consensus_now,
        consensus_last8h_min_f=consensus_8h_min,
        consensus_last8h_max_f=consensus_8h_max,

        consensus_tmr_min_f=consensus_tmr_min,
        consensus_tmr_max_f=consensus_tmr_max,

        threshold_f=threshold,
        p_over=p_over,
        market_price=market,
        edge=edge,
        stake=stake,
    )

def run_once_text() -> str:
    rr = run_once_struct()
    def ff(x): return "N/A" if x is None else f"{x:.1f}F"

    lines = []
    lines.append(f"Generated: {rr.generated_at_utc}")
    lines.append(f"Location: {rr.lat},{rr.lon}")
    lines.append(f"Tomorrow (NY): {rr.ny_date_tomorrow}")

    lines.append("\nNOW + last 8h:")
    lines.append(f"Consensus now/min/max: {ff(rr.consensus_now_f)} / {ff(rr.consensus_last8h_min_f)} / {ff(rr.consensus_last8h_max_f)}")

    for s in rr.sources:
        lines.append(f"\n[{s.src}] status={s.status} inserted={s.rows_inserted}")
        lines.append(f"Now / 8h MIN / 8h MAX: {ff(s.now_f)} / {ff(s.last8h_min_f)} / {ff(s.last8h_max_f)}")
        lines.append(f"Tomorrow NY MIN/MAX: {ff(s.tmr_min_f)} / {ff(s.tmr_max_f)}")

    if rr.consensus_tmr_min_f is not None and rr.consensus_tmr_max_f is not None:
        lines.append(f"\nTomorrow consensus MIN/MAX: {ff(rr.consensus_tmr_min_f)} / {ff(rr.consensus_tmr_max_f)}")
    else:
        lines.append("\nTomorrow consensus MIN/MAX: N/A")

    if rr.p_over is not None:
        lines.append(f"P(TMAX > {rr.threshold_f:.0f}F): {rr.p_over*100:.2f}%")
        lines.append(f"Market YES: {rr.market_price*100:.2f}% | Edge: {rr.edge*100:.2f}%")
        lines.append(f"Stake (25% Kelly): {rr.stake*100:.2f}%")

    if rr.removed_outliers:
        lines.append(f"\nOutliers removed: {rr.removed_outliers}")

    if rr.notes:
        lines.append("\nNotes:")
        lines.extend([f"- {x}" for x in rr.notes])

    return "\n".join(lines)

if __name__ == "__main__":
    print(run_once_text())
