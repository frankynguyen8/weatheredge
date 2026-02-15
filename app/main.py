import sqlite3, json, time, argparse
from datetime import datetime, timedelta, timezone
import requests
import numpy as np
from zoneinfo import ZoneInfo

LAT = 40.78858
LON = -73.9661
NY = ZoneInfo("America/New_York")

TOMORROW_KEY = "iqAjw2K5LNqSOLQfBbYASQ1izFV51xjy"
import os
DB_PATH = os.getenv("DB_PATH", "/var/data/weatheredge.sqlite")

def db():
    # ✅ Ensure folder exists (Render Disk)
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
    fetched = datetime.now(timezone.utc).isoformat()
    con.executemany(
        "INSERT INTO hourly_forecast(source,time_utc,temp_f,fetched_at_utc) VALUES (?,?,?,?)",
        [(source, t, tf, fetched) for (t, tf) in rows]
    )
    con.commit()

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

def fetch_open_meteo(horizon_hours=48):
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        "&hourly=temperature_2m"
        "&temperature_unit=fahrenheit"
        "&timezone=UTC"
    )
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    j = r.json()

    now = datetime.now(timezone.utc)
    end = now + timedelta(hours=horizon_hours)

    rows = []
    for t, tf in zip(j["hourly"]["time"], j["hourly"]["temperature_2m"]):
        dt = datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
        if now <= dt <= end:
            rows.append((dt.isoformat(), float(tf)))
    return rows

def fetch_weather_gov(horizon_hours=48):
    headers = {"User-Agent": "WeatherEdge (long22nguyenhuu@icloud.com)", "Accept":"application/geo+json"}
    p = requests.get(f"https://api.weather.gov/points/{LAT},{LON}", headers=headers, timeout=20)
    p.raise_for_status()
    forecast_url = p.json()["properties"]["forecastHourly"]

    r = requests.get(forecast_url, headers=headers, timeout=20)
    r.raise_for_status()
    periods = r.json()["properties"]["periods"]

    now = datetime.now(timezone.utc)
    end = now + timedelta(hours=horizon_hours)

    rows = []
    for it in periods:
        dt = datetime.fromisoformat(it["startTime"]).astimezone(timezone.utc)
        if now <= dt <= end:
            rows.append((dt.isoformat(), float(it["temperature"])))
    return rows

def fetch_tomorrow_io(horizon_hours=48):
    if TOMORROW_KEY.startswith("PUT_"):
        print("Tomorrow.io key not set -> skipping Tomorrow.io.")
        return []

    url = (
        "https://api.tomorrow.io/v4/weather/forecast"
        f"?location={LAT},{LON}"
        "&fields=temperature"
        "&timesteps=1h"
        "&units=imperial"
        f"&apikey={TOMORROW_KEY}"
    )
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    j = r.json()

    now = datetime.now(timezone.utc)
    end = now + timedelta(hours=horizon_hours)

    rows = []
    for it in j["timelines"]["hourly"]:
        dt = datetime.fromisoformat(it["time"].replace("Z", "+00:00")).astimezone(timezone.utc)
        if now <= dt <= end:
            rows.append((dt.isoformat(), float(it["values"]["temperature"])))
    return rows

def next_n_hours(series, n=18):
    now = datetime.now(timezone.utc)
    out = []
    for dt, tf in series:
        if dt >= now:
            out.append((dt, tf))
        if len(out) >= n:
            break
    return out

def max_temp_tomorrow_ny(series):
    target = (datetime.now(NY) + timedelta(days=1)).date()
    mx = None
    for dt_utc, tf in series:
        dt_ny = dt_utc.astimezone(NY)
        if dt_ny.date() == target:
            mx = tf if mx is None else max(mx, tf)
    return mx

def remove_outliers_iqr(values_dict):
    vals = np.array(list(values_dict.values()), dtype=float)
    if len(vals) < 3:
        return values_dict
    q1 = np.percentile(vals, 25)
    q3 = np.percentile(vals, 75)
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    filtered = {k:v for k,v in values_dict.items() if lo <= v <= hi}
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

def monte_carlo_prob_over(threshold_f, mean_f, sigma_f, sims=50000, seed=7):
    rng = np.random.default_rng(seed)
    draws = rng.normal(loc=mean_f, scale=sigma_f, size=sims)
    return float(np.mean(draws > threshold_f))

def read_market_price():
    try:
        with open("/app/market.json","r",encoding="utf-8") as f:
            j = json.load(f)
        return float(j["market_price_yes_over_48"])
    except Exception:
        return 0.17

def fractional_kelly(p, price, fraction=0.25):
    if price <= 0 or price >= 1:
        return 0.0
    raw = (p - price) / (1 - price)
    return max(0.0, raw) * fraction

def run_once():
    lines = []

    con = db()

    lines.append("Fetching Open-Meteo...")
    insert_rows(con, "open_meteo", fetch_open_meteo())

    lines.append("Fetching Weather.gov...")
    insert_rows(con, "weather_gov", fetch_weather_gov())

    lines.append("Fetching Tomorrow.io...")
    insert_rows(con, "tomorrow_io", fetch_tomorrow_io())

    by_source = load_recent(con, hours=6)
    lines.append(f"Sources: {list(by_source.keys())}")

    for src, series in by_source.items():
        nxt = next_n_hours(series, 18)
        lines.append(f"\n--- {src}: next {len(nxt)} hours (UTC) ---")
        for dt, tf in nxt:
            lines.append(f"{dt.isoformat()}  {tf:.1f}F")

    mean, sigma, per, removed = consensus_tomorrow_max(by_source)
    if mean is None:
        lines.append("\nNo tomorrow data within current horizon.")
        return "\n".join(lines)

    lines.append("\nTomorrow (NY) max per source:")
    for src, mx in per.items():
        lines.append(f"  {src}: {mx:.2f}F")
    if removed:
        lines.append(f"Outliers removed: {removed}")

    base_sigma = 2.0
    sigma_total = float(np.sqrt(base_sigma**2 + sigma**2))

    lines.append(f"\nConsensus tomorrow max (NY): {mean:.2f}F")
    lines.append(f"Model spread sigma: {sigma:.2f}F | Base sigma: {base_sigma:.2f}F | Total sigma: {sigma_total:.2f}F")

    threshold = 48.0
    p_over = monte_carlo_prob_over(threshold, mean, sigma_total)
    lines.append(f"\nMonte Carlo P(TMAX > {threshold:.0f}F): {p_over*100:.2f}%")

    market_price = read_market_price()
    edge = p_over - market_price
    kelly = fractional_kelly(p_over, market_price, fraction=0.25)

    lines.append(f"\nMarket price YES(>{threshold:.0f}F): {market_price*100:.2f}%")
    lines.append(f"Model prob: {p_over*100:.2f}% | Edge: {edge*100:.2f}%")
    lines.append(f"Suggested stake (25% Kelly): {kelly*100:.2f}% of bankroll")

    return "\n".join(lines)

def main():
    run_once()

def home():
    return "WeatherEdge is running! Try /api/run"

@app.get("/api/run")
def api_run():
    try:
        out = run_once()   # run_once() phải trả về string output

        # Nếu muốn JSON thì thêm ?format=json
        if request.args.get("format") == "json":
            return jsonify(ok=True, output=out.splitlines())

        # Mặc định trả text/plain để đọc đẹp
        return Response(out + "\n", mimetype="text/plain; charset=utf-8")

    except Exception as e:
        return jsonify(ok=False, error=str(e))
