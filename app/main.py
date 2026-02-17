import os
import json
import time
import sqlite3
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import requests

LAT = float(os.getenv("LAT", "40.78858"))
LON = float(os.getenv("LON", "-73.9661"))
NY = ZoneInfo("America/New_York")

DB_PATH = os.getenv("DB_PATH", "/var/data/weatheredge.sqlite")
TOMORROW_KEY = os.getenv("TOMORROW_KEY", "")

# -------------------- HTTP SAFE GET --------------------

def safe_get(url: str, headers=None, timeout=15, retries=3, backoff=1.5):
    """
    - Retry 429, 5xx, network errors
    - Return Response or None (never raise)
    """
    headers = headers or {}
    last_err = None

    for i in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)

            # 429 Too Many Requests
            if r.status_code == 429:
                time.sleep(backoff * (i + 1))
                continue

            # 5xx server errors
            if 500 <= r.status_code <= 599:
                time.sleep(backoff * (i + 1))
                continue

            return r
        except requests.RequestException as e:
            last_err = e
            time.sleep(backoff * (i + 1))

    return None

# -------------------- DB --------------------

def db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS hourly_forecast (
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

# -------------------- TIME HELPERS --------------------

def horizon_filter(dt: datetime, horizon_hours=48):
    now = datetime.now(timezone.utc)
    end = now + timedelta(hours=horizon_hours)
    return now <= dt <= end

def next_n_hours(series, n=18):
    now = datetime.now(timezone.utc)
    out = [(dt, tf) for dt, tf in series if dt >= now]
    return out[:n]

def max_temp_tomorrow_ny(series):
    target = (datetime.now(NY) + timedelta(days=1)).date()
    mx = None
    for dt_utc, tf in series:
        dt_ny = dt_utc.astimezone(NY)
        if dt_ny.date() == target:
            mx = tf if mx is None else max(mx, tf)
    return mx

# -------------------- FETCHERS (SAFE) --------------------

def fetch_open_meteo(horizon_hours=48):
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        "&hourly=temperature_2m"
        "&temperature_unit=fahrenheit"
        "&timezone=UTC"
    )
    r = safe_get(url, timeout=15, retries=3, backoff=1.8)
    if not r:
        return []

    try:
        j = r.json()
        rows = []
        for t, tf in zip(j["hourly"]["time"], j["hourly"]["temperature_2m"]):
            dt = datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
            if horizon_filter(dt, horizon_hours):
                rows.append((dt.isoformat(), float(tf)))
        return rows
    except Exception:
        return []

def fetch_weather_gov(horizon_hours=48):
    headers = {
        "User-Agent": "WeatherEdge (contact: long22nguyenhuu@icloud.com)",
        "Accept": "application/geo+json",
    }

    p = safe_get(f"https://api.weather.gov/points/{LAT},{LON}", headers=headers, timeout=15, retries=3)
    if not p:
        return []

    try:
        forecast_url = p.json()["properties"]["forecastHourly"]
    except Exception:
        return []

    r = safe_get(forecast_url, headers=headers, timeout=15, retries=3)
    if not r:
        return []

    try:
        periods = r.json()["properties"]["periods"]
        rows = []
        for it in periods:
            dt = datetime.fromisoformat(it["startTime"]).astimezone(timezone.utc)
            if horizon_filter(dt, horizon_hours):
                rows.append((dt.isoformat(), float(it["temperature"])))
        return rows
    except Exception:
        return []

def fetch_tomorrow_io(horizon_hours=48):
    if not TOMORROW_KEY:
        return []

    url = (
        "https://api.tomorrow.io/v4/weather/forecast"
        f"?location={LAT},{LON}"
        "&fields=temperature"
        "&timesteps=1h"
        "&units=imperial"
        f"&apikey={TOMORROW_KEY}"
    )

    r = safe_get(url, timeout=15, retries=3, backoff=1.8)
    if not r:
        return []

    try:
        j = r.json()
        rows = []
        for it in j["timelines"]["hourly"]:
            dt = datetime.fromisoformat(it["time"].replace("Z", "+00:00")).astimezone(timezone.utc)
            if horizon_filter(dt, horizon_hours):
                rows.append((dt.isoformat(), float(it["values"]["temperature"])))
        return rows
    except Exception:
        return []

# -------------------- CONSENSUS + BETTING --------------------

def remove_outliers_iqr(values_dict):
    vals = np.array(list(values_dict.values()), dtype=float)
    if len(vals) < 3:
        return values_dict
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    filtered = {k: v for k, v in values_dict.items() if lo <= v <= hi}
    return filtered if len(filtered) >= 2 else values_dict

def consensus_tomorrow_max(by_source):
    per = {}
    for src, series in by_source.items():
        mx = max_temp_tomorrow_ny(series)
        if mx is not None:
            per[src] = mx

    if not per:
        return None, None, None, None

    per_filtered = remove_outliers_iqr(per)
    removed = sorted(set(per.keys()) - set(per_filtered.keys()))

    vals = list(per_filtered.values())
    mean = float(np.mean(vals))
    spread = float(np.std(vals)) if len(vals) > 1 else 2.0
    return mean, spread, per, removed

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

# -------------------- RUN --------------------

def run_once() -> str:
    lines = []
    con = db()

    # Fetch (never crash)
    n1 = insert_rows(con, "open_meteo", fetch_open_meteo())
    n2 = insert_rows(con, "weather_gov", fetch_weather_gov())
    n3 = insert_rows(con, "tomorrow_io", fetch_tomorrow_io())

    lines.append(f"Inserted rows: open_meteo={n1}, weather_gov={n2}, tomorrow_io={n3}")

    by_source = load_recent(con, hours=6)
    lines.append(f"Sources available: {sorted(by_source.keys())}")

    # Optional: show next few hours (keep short)
    for src, series in by_source.items():
        nxt = next_n_hours(series, 8)
        if nxt:
            lines.append(f"\n{src} next hours (UTC):")
            for dt, tf in nxt:
                lines.append(f"  {dt.isoformat()}  {tf:.1f}F")

    mean, sigma, per, removed = consensus_tomorrow_max(by_source)
    if mean is None:
        lines.append("\nNo tomorrow data within horizon (providers may be rate-limiting).")
        return "\n".join(lines)

    lines.append("\nTomorrow (NY) max per source:")
    for src, mx in per.items():
        lines.append(f"  {src}: {mx:.2f}F")
    if removed:
        lines.append(f"Outliers removed: {removed}")

    base_sigma = 2.0
    sigma_total = float(np.sqrt(base_sigma**2 + sigma**2))

    threshold = 48.0
    p_over = monte_carlo_prob_over(threshold, mean, sigma_total)

    market_price = read_market_price()
    edge = p_over - market_price
    kelly = fractional_kelly(p_over, market_price, fraction=0.25)

    lines.append(f"\nConsensus tomorrow max (NY): {mean:.2f}F")
    lines.append(f"Total sigma: {sigma_total:.2f}F")
    lines.append(f"P(TMAX > {threshold:.0f}F): {p_over*100:.2f}%")
    lines.append(f"Market YES: {market_price*100:.2f}% | Edge: {edge*100:.2f}%")
    lines.append(f"Stake (25% Kelly): {kelly*100:.2f}% of bankroll")

    return "\n".join(lines)

def run_once_text() -> str:
    # helper for API
    return run_once()

if __name__ == "__main__":
    print(run_once())
