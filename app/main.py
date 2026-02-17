import os
import json
import time
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import requests

# ---------------- CONFIG ----------------

DB_PATH = os.getenv("DB_PATH", "/var/data/weatheredge.sqlite")
TOMORROW_KEY = os.getenv("TOMORROW_KEY", "").strip()

# Default location (still used if user doesn't provide query)
DEFAULT_LAT = float(os.getenv("LAT", "40.78858"))
DEFAULT_LON = float(os.getenv("LON", "-73.9661"))

EXPECTED_SOURCES = ["open_meteo", "weather_gov", "tomorrow_io"]

# Kalshi (optional - keeps your previous integration style)
KALSHI_MARKET_TICKER = os.getenv("KALSHI_MARKET_TICKER", "").strip()
KALSHI_BASE_URL = os.getenv("KALSHI_BASE_URL", "https://api.elections.kalshi.com").rstrip("/")

# ---------------- HTTP SAFE ----------------

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

# ---------------- GEOCODE (USA) ----------------
# Uses Open-Meteo geocoding API (free)

def geocode_us(query: str, count: int = 5):
    """
    Returns list of candidates in US:
      {name, admin1, latitude, longitude, timezone, country_code}
    """
    q = (query or "").strip()
    if not q:
        return []

    url = (
        "https://geocoding-api.open-meteo.com/v1/search"
        f"?name={requests.utils.quote(q)}&count={count}&language=en&format=json"
    )
    r = safe_get(url, timeout=15, retries=2)
    if not r or r.status_code != 200:
        return []

    try:
        j = r.json()
        out = []
        for it in j.get("results", []) or []:
            if it.get("country_code") != "US":
                continue
            out.append({
                "name": it.get("name"),
                "admin1": it.get("admin1"),
                "latitude": float(it.get("latitude")),
                "longitude": float(it.get("longitude")),
                "timezone": it.get("timezone"),
                "country_code": it.get("country_code"),
            })
        return out
    except Exception:
        return []

def resolve_location(q: str | None, lat: float | None, lon: float | None):
    """
    Priority:
      - if lat/lon provided -> use them
      - else if q provided -> geocode first match in US
      - else -> default env location
    Returns: (lat, lon, display_name, tz_name)
    """
    if lat is not None and lon is not None:
        # no display name; timezone can be fetched from open-meteo later; keep UTC fallback
        return float(lat), float(lon), f"{lat:.4f},{lon:.4f}", "UTC"

    q = (q or "").strip()
    if q:
        cands = geocode_us(q, count=5)
        if cands:
            top = cands[0]
            name = top["name"]
            admin1 = top["admin1"]
            disp = f"{name}, {admin1}" if admin1 else name
            tz = top.get("timezone") or "UTC"
            return top["latitude"], top["longitude"], disp, tz

    return DEFAULT_LAT, DEFAULT_LON, f"{DEFAULT_LAT:.4f},{DEFAULT_LON:.4f}", "UTC"

# ---------------- DB ----------------

def ensure_schema(con: sqlite3.Connection):
    con.execute("""
        CREATE TABLE IF NOT EXISTS hourly_forecast(
            source TEXT,
            lat REAL,
            lon REAL,
            time_utc TEXT,
            temp_f REAL,
            fetched_at_utc TEXT
        )
    """)
    con.commit()

def db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    ensure_schema(con)
    return con

def insert_rows(con, source, lat, lon, rows):
    if not rows:
        return 0
    fetched = datetime.now(timezone.utc).isoformat()
    con.executemany(
        "INSERT INTO hourly_forecast(source,lat,lon,time_utc,temp_f,fetched_at_utc) VALUES (?,?,?,?,?,?)",
        [(source, float(lat), float(lon), t, tf, fetched) for (t, tf) in rows]
    )
    con.commit()
    return len(rows)

def load_recent(con, lat, lon, hours=6):
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    cur = con.execute(
        """
        SELECT source, time_utc, temp_f
        FROM hourly_forecast
        WHERE fetched_at_utc >= ?
          AND ABS(lat - ?) < 0.00001
          AND ABS(lon - ?) < 0.00001
        """,
        (cutoff, float(lat), float(lon))
    )
    by_source = {}
    for src, t, tf in cur.fetchall():
        by_source.setdefault(src, []).append((datetime.fromisoformat(t), float(tf)))
    for src in by_source:
        by_source[src].sort(key=lambda x: x[0])
    return by_source

# ---------------- TIME HELPERS ----------------

def in_horizon(dt: datetime, horizon_hours=48):
    now = datetime.now(timezone.utc)
    end = now + timedelta(hours=horizon_hours)
    return now <= dt <= end

def day_minmax_local(series, local_tz: ZoneInfo, day_offset=1):
    """
    Tomorrow based on local timezone for the chosen location
    """
    target = (datetime.now(local_tz) + timedelta(days=day_offset)).date()
    vals = []
    for dt_utc, tf in series:
        dt_loc = dt_utc.astimezone(local_tz)
        if dt_loc.date() == target:
            vals.append(float(tf))
    if not vals:
        return None, None
    return float(min(vals)), float(max(vals))

def last_hours_now_minmax(series, hours=8):
    """
    NOW = latest datapoint <= now
    MIN/MAX = last N hours in UTC window
    """
    if not series:
        return None, None, None

    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours)

    window_vals = [tf for dt, tf in series if start <= dt <= now]
    mn = float(min(window_vals)) if window_vals else None
    mx = float(max(window_vals)) if window_vals else None

    past = [(dt, tf) for dt, tf in series if dt <= now]
    now_tf = float(past[-1][1]) if past else None
    return now_tf, mn, mx

# ---------------- FETCHERS ----------------

def fetch_open_meteo(lat, lon, horizon_hours=48):
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m"
        "&temperature_unit=fahrenheit"
        "&timezone=UTC"
    )
    r = safe_get(url)
    if not r or r.status_code != 200:
        return [], None

    try:
        j = r.json()
        tz_name = j.get("timezone")  # usually "GMT" since we requested UTC; keep for safety
        rows = []
        for t, tf in zip(j["hourly"]["time"], j["hourly"]["temperature_2m"]):
            dt = datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
            if in_horizon(dt, horizon_hours):
                rows.append((dt.isoformat(), float(tf)))
        return rows, tz_name
    except Exception:
        return [], None

def fetch_weather_gov(lat, lon, horizon_hours=48):
    """
    Only works in USA coverage; for non-covered points it will fail => [].
    """
    headers = {"User-Agent": "WeatherEdge", "Accept": "application/geo+json"}
    p = safe_get(f"https://api.weather.gov/points/{lat},{lon}", headers=headers)
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

def fetch_tomorrow_io(lat, lon, horizon_hours=48):
    if not TOMORROW_KEY:
        return []

    url = "https://api.tomorrow.io/v4/timelines"
    payload = {
        "location": f"{lat},{lon}",
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

# ---------------- KALSHI MARKET (optional) ----------------

def fetch_kalshi_yes_price(market_ticker: str):
    if not market_ticker:
        return None, None

    url = f"{KALSHI_BASE_URL}/trade-api/v2/markets/{market_ticker}"
    r = safe_get(url, timeout=15, retries=2)
    if not r or r.status_code != 200:
        return None, None

    try:
        j = r.json()
        m = j.get("market", {})
        if m.get("yes_ask_dollars") is not None:
            return float(m["yes_ask_dollars"]), "yes_ask_dollars"
        if m.get("last_price_dollars") is not None:
            return float(m["last_price_dollars"]), "last_price_dollars"
        return None, None
    except Exception:
        return None, None

def read_market_price():
    # 1) Kalshi
    p, used = fetch_kalshi_yes_price(KALSHI_MARKET_TICKER)
    if p is not None:
        return p, "kalshi", {"ticker": KALSHI_MARKET_TICKER, "field": used}

    # 2) local json
    try:
        with open("/app/market.json", "r", encoding="utf-8") as f:
            j = json.load(f)
        return float(j["market_price_yes_over_48"]), "local_json", {"path": "/app/market.json"}
    except Exception:
        pass

    # 3) env fallback
    return float(os.getenv("MARKET_PRICE_YES_OVER_48", "0.17")), "env_fallback", {"env": "MARKET_PRICE_YES_OVER_48"}

# ---------------- STATS & BETTING ----------------

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

def fractional_kelly(p, price, fraction=0.25):
    if not (0 < price < 1):
        return 0.0
    raw = (p - price) / (1 - price)
    return max(0.0, raw) * fraction

# ---------------- MODELS ----------------

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

    location_name: str
    timezone_name: str
    lat: float
    lon: float
    local_tomorrow_date: str

    sources: list
    removed_outliers: list
    notes: list

    consensus_now_f: float | None
    consensus_last8h_min_f: float | None
    consensus_last8h_max_f: float | None

    consensus_tmr_min_f: float | None
    consensus_tmr_max_f: float | None

    # betting
    threshold_f: float
    p_over: float | None
    market_price: float | None
    market_source: str
    market_meta: dict
    edge: float | None
    stake: float | None

# ---------------- RUNNER ----------------

def run_once_struct(q: str | None = None, lat: float | None = None, lon: float | None = None) -> RunResult:
    lat0, lon0, loc_name, tz_guess = resolve_location(q, lat, lon)

    # Determine local timezone
    # If we got tz from geocode -> use it; else keep UTC
    try:
        local_tz = ZoneInfo(tz_guess) if tz_guess else ZoneInfo("UTC")
        tz_name = local_tz.key
    except Exception:
        local_tz = ZoneInfo("UTC")
        tz_name = "UTC"

    con = db()
    notes = []

    # Fetch + insert
    om_rows, _tz_from_om = fetch_open_meteo(lat0, lon0)
    n1 = insert_rows(con, "open_meteo", lat0, lon0, om_rows)

    wg_rows = fetch_weather_gov(lat0, lon0)
    n2 = insert_rows(con, "weather_gov", lat0, lon0, wg_rows)

    tm_rows = fetch_tomorrow_io(lat0, lon0)
    n3 = insert_rows(con, "tomorrow_io", lat0, lon0, tm_rows)

    inserts = {"open_meteo": n1, "weather_gov": n2, "tomorrow_io": n3}

    by_source = load_recent(con, lat0, lon0, hours=6)

    per_now = {}
    per_8h_min = {}
    per_8h_max = {}
    per_tmr_min = {}
    per_tmr_max = {}

    src_summaries = []

    for src in EXPECTED_SOURCES:
        series = by_source.get(src, [])
        if src == "tomorrow_io" and not TOMORROW_KEY:
            status = "NO_KEY"
        elif not series:
            status = "EMPTY"
        else:
            status = "OK"

        now_f, mn8, mx8 = last_hours_now_minmax(series, hours=8)
        tmin, tmax = day_minmax_local(series, local_tz, day_offset=1)

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

    # consensus NOW + 8h
    consensus_now = None
    consensus_8h_min = None
    consensus_8h_max = None

    if per_now:
        now_filtered = remove_outliers_iqr(per_now)
        removed = sorted(set(removed) | (set(per_now.keys()) - set(now_filtered.keys())))
        consensus_now = float(np.mean(list(now_filtered.values())))
    else:
        notes.append("No NOW datapoints (last 6h cache) for this location.")

    if per_8h_min:
        min_filtered = remove_outliers_iqr(per_8h_min)
        removed = sorted(set(removed) | (set(per_8h_min.keys()) - set(min_filtered.keys())))
        consensus_8h_min = float(np.mean(list(min_filtered.values())))
    else:
        notes.append("No last-8h MIN values for this location.")

    if per_8h_max:
        max_filtered = remove_outliers_iqr(per_8h_max)
        removed = sorted(set(removed) | (set(per_8h_max.keys()) - set(max_filtered.keys())))
        consensus_8h_max = float(np.mean(list(max_filtered.values())))
    else:
        notes.append("No last-8h MAX values for this location.")

    # tomorrow consensus
    consensus_tmr_min = consensus_tmr_max = None
    if per_tmr_min and per_tmr_max:
        tmr_min_filtered = remove_outliers_iqr(per_tmr_min)
        tmr_max_filtered = remove_outliers_iqr(per_tmr_max)
        removed = sorted(set(removed) |
                         (set(per_tmr_min.keys()) - set(tmr_min_filtered.keys())) |
                         (set(per_tmr_max.keys()) - set(tmr_max_filtered.keys())))
        consensus_tmr_min = float(np.mean(list(tmr_min_filtered.values())))
        consensus_tmr_max = float(np.mean(list(tmr_max_filtered.values())))
    else:
        notes.append("Not enough tomorrow data to compute MIN/MAX consensus.")

    # market price (kalshi or fallback)
    market_price, market_source, market_meta = read_market_price()

    # betting
    threshold = 48.0
    p_over = edge = stake = None
    if consensus_tmr_max is not None:
        vals = list(per_tmr_max.values())
        spread = float(np.std(vals)) if len(vals) > 1 else 2.0
        base_sigma = 2.0
        sigma_total = float(np.sqrt(base_sigma**2 + spread**2))

        p_over = monte_carlo_prob_over(threshold, consensus_tmr_max, sigma_total)
        edge = p_over - market_price
        stake = fractional_kelly(p_over, market_price, fraction=0.25)

    local_tomorrow = (datetime.now(local_tz) + timedelta(days=1)).date().isoformat()

    # quick note why only 1 source appears in practice
    notes.append(f"Inserted rows: open_meteo={n1}, weather_gov={n2}, tomorrow_io={n3}")

    return RunResult(
        ok=True,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),

        location_name=loc_name,
        timezone_name=tz_name,
        lat=float(lat0),
        lon=float(lon0),
        local_tomorrow_date=local_tomorrow,

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
        market_price=market_price,
        market_source=market_source,
        market_meta=market_meta,
        edge=edge,
        stake=stake,
    )

def run_once_text(q: str | None = None, lat: float | None = None, lon: float | None = None) -> str:
    rr = run_once_struct(q=q, lat=lat, lon=lon)
    def ff(x): return "N/A" if x is None else f"{x:.1f}F"

    lines = []
    lines.append(f"Generated: {rr.generated_at_utc}")
    lines.append(f"Location: {rr.location_name} ({rr.lat:.4f},{rr.lon:.4f}) TZ={rr.timezone_name}")
    lines.append(f"Tomorrow (local): {rr.local_tomorrow_date}")
    lines.append(f"Market: {rr.market_source} price={rr.market_price:.4f}")

    lines.append("\nNOW + last 8h:")
    lines.append(f"Consensus now/min/max: {ff(rr.consensus_now_f)} / {ff(rr.consensus_last8h_min_f)} / {ff(rr.consensus_last8h_max_f)}")

    for s in rr.sources:
        lines.append(f"\n[{s.src}] status={s.status} inserted={s.rows_inserted}")
        lines.append(f"Now / 8h MIN / 8h MAX: {ff(s.now_f)} / {ff(s.last8h_min_f)} / {ff(s.last8h_max_f)}")
        lines.append(f"Tomorrow MIN/MAX: {ff(s.tmr_min_f)} / {ff(s.tmr_max_f)}")

    if rr.p_over is not None:
        lines.append(f"\nP(TMAX > {rr.threshold_f:.0f}F): {rr.p_over*100:.2f}%")
        lines.append(f"Edge: {(rr.edge*100):.2f}% | Stake(25% Kelly): {(rr.stake*100):.2f}%")

    if rr.removed_outliers:
        lines.append(f"\nOutliers removed: {rr.removed_outliers}")

    if rr.notes:
        lines.append("\nNotes:")
        lines.extend([f"- {x}" for x in rr.notes])

    return "\n".join(lines)

if __name__ == "__main__":
    print(run_once_text())
