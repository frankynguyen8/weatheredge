import os
import json
import time
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

# ================= CONFIG =================

DB_PATH = os.getenv("DB_PATH", "/var/data/weatheredge.sqlite")

DEFAULT_LAT = float(os.getenv("LAT", "40.78858"))
DEFAULT_LON = float(os.getenv("LON", "-73.9661"))

TOMORROW_KEY = os.getenv("TOMORROW_KEY", "").strip()

EXPECTED_SOURCES = ["open_meteo", "weather_gov", "tomorrow_io"]

# Optional Kalshi market data (public)
KALSHI_MARKET_TICKER = os.getenv("KALSHI_MARKET_TICKER", "").strip()
KALSHI_BASE_URL = os.getenv("KALSHI_BASE_URL", "https://api.elections.kalshi.com").rstrip("/")

# ================= HTTP SAFE =================

def safe_get(url, headers=None, timeout=15, retries=3, backoff=1.8):
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

def safe_post(url, headers=None, json_body=None, timeout=15, retries=3, backoff=1.8):
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

# ================= SIMPLE STATS (NO NUMPY) =================

def mean(xs):
    xs = list(xs)
    return (sum(xs) / len(xs)) if xs else None

def stdev(xs):
    xs = list(xs)
    n = len(xs)
    if n <= 1:
        return 0.0
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return var ** 0.5

def quantile(sorted_xs, q):
    """Linear interpolation quantile. q in [0,1]. sorted_xs must be sorted."""
    n = len(sorted_xs)
    if n == 0:
        return None
    if n == 1:
        return float(sorted_xs[0])
    pos = (n - 1) * q
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(sorted_xs[lo] * (1 - frac) + sorted_xs[hi] * frac)

def remove_outliers_iqr(values_dict):
    """
    values_dict: {source: value}
    returns filtered dict
    """
    if not values_dict or len(values_dict) < 3:
        return values_dict

    vals = sorted(float(v) for v in values_dict.values())
    q1 = quantile(vals, 0.25)
    q3 = quantile(vals, 0.75)
    if q1 is None or q3 is None:
        return values_dict

    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    filtered = {k: v for k, v in values_dict.items() if lo <= float(v) <= hi}
    return filtered if len(filtered) >= 2 else values_dict

def monte_carlo_prob_over(threshold_f, mean_f, sigma_f, sims=20000, seed=7):
    """
    Normal draws without numpy (Box-Muller).
    """
    import random, math
    random.seed(seed)
    sigma = max(0.1, float(sigma_f))
    count = 0
    for _ in range(sims):
        # Box-Muller
        u1 = max(1e-12, random.random())
        u2 = random.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        x = float(mean_f) + sigma * z
        if x > threshold_f:
            count += 1
    return count / sims

def fractional_kelly(p, price, fraction=0.25):
    if p is None or price is None:
        return 0.0
    if not (0 < price < 1):
        return 0.0
    raw = (p - price) / (1 - price)
    return max(0.0, raw) * fraction

# ================= GEO (USA) =================

def geocode_us(query, count=5):
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
        for it in (j.get("results") or []):
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

def safe_zoneinfo(tz_name):
    try:
        if tz_name:
            tz = ZoneInfo(tz_name)
            return tz, tz.key
    except Exception:
        pass
    tz = ZoneInfo("UTC")
    return tz, "UTC"

def resolve_location(q=None, lat=None, lon=None):
    if lat is not None and lon is not None:
        return float(lat), float(lon), f"{float(lat):.4f},{float(lon):.4f}", "UTC"

    q = (q or "").strip()
    if q:
        cands = geocode_us(q, count=5)
        if cands:
            top = cands[0]
            name = top.get("name") or "Unknown"
            admin1 = top.get("admin1") or ""
            disp = f"{name}, {admin1}".strip().strip(",")
            tz = top.get("timezone") or "UTC"
            return top["latitude"], top["longitude"], disp, tz

    return DEFAULT_LAT, DEFAULT_LON, f"{DEFAULT_LAT:.4f},{DEFAULT_LON:.4f}", "UTC"

# ================= DB (MIGRATION SAFE) =================

def table_columns(con, table):
    cur = con.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cur.fetchall()]

def migrate_schema(con):
    cols = set(table_columns(con, "hourly_forecast"))

    # Create if missing (old deployments)
    if not cols:
        con.execute("""
            CREATE TABLE IF NOT EXISTS hourly_forecast(
                source TEXT,
                time_utc TEXT,
                temp_f REAL,
                fetched_at_utc TEXT
            )
        """)
        con.commit()
        cols = set(table_columns(con, "hourly_forecast"))

    altered = False
    if "lat" not in cols:
        con.execute("ALTER TABLE hourly_forecast ADD COLUMN lat REAL")
        altered = True
    if "lon" not in cols:
        con.execute("ALTER TABLE hourly_forecast ADD COLUMN lon REAL")
        altered = True
    if altered:
        con.commit()

    # Backfill NULL lat/lon rows to default to avoid select filters returning empty
    con.execute(
        "UPDATE hourly_forecast SET lat=?, lon=? WHERE lat IS NULL OR lon IS NULL",
        (DEFAULT_LAT, DEFAULT_LON),
    )
    con.commit()

def db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    migrate_schema(con)
    return con

def insert_rows(con, source, lat, lon, rows):
    if not rows:
        return 0
    fetched = datetime.now(timezone.utc).isoformat()
    con.executemany(
        "INSERT INTO hourly_forecast(source,lat,lon,time_utc,temp_f,fetched_at_utc) VALUES (?,?,?,?,?,?)",
        [(source, float(lat), float(lon), t, float(tf), fetched) for (t, tf) in rows]
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

# ================= TIME HELPERS =================

def in_horizon(dt, horizon_hours=48):
    now = datetime.now(timezone.utc)
    end = now + timedelta(hours=horizon_hours)
    return now <= dt <= end

def last_hours_now_minmax(series, hours=8):
    if not series:
        return None, None, None

    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours)

    window = [tf for dt, tf in series if start <= dt <= now]
    mn = min(window) if window else None
    mx = max(window) if window else None

    past = [(dt, tf) for dt, tf in series if dt <= now]
    now_tf = past[-1][1] if past else None
    return now_tf, mn, mx

def day_minmax_local(series, local_tz, day_offset=1):
    target = (datetime.now(local_tz) + timedelta(days=day_offset)).date()
    vals = []
    for dt_utc, tf in series:
        if dt_utc.astimezone(local_tz).date() == target:
            vals.append(float(tf))
    if not vals:
        return None, None
    return min(vals), max(vals)

# ================= FETCHERS =================

def fetch_open_meteo(lat, lon, horizon_hours=48):
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m"
        "&temperature_unit=fahrenheit"
        "&timezone=UTC"
    )
    r = safe_get(url, timeout=20, retries=2)
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

def fetch_weather_gov(lat, lon, horizon_hours=48):
    headers = {
        "User-Agent": "WeatherEdge (contact: long22nguyenhuu@icloud.com)",
        "Accept": "application/geo+json",
    }
    p = safe_get(f"https://api.weather.gov/points/{lat},{lon}", headers=headers, timeout=20, retries=2)
    if not p or p.status_code != 200:
        return []
    try:
        forecast_url = p.json()["properties"]["forecastHourly"]
    except Exception:
        return []

    r = safe_get(forecast_url, headers=headers, timeout=20, retries=2)
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

    r = safe_post(url, headers=headers, json_body=payload, timeout=20, retries=2)
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

# ================= KALSHI MARKET (PUBLIC) =================

def fetch_kalshi_yes_price(ticker):
    if not ticker:
        return None, None
    url = f"{KALSHI_BASE_URL}/trade-api/v2/markets/{ticker}"
    r = safe_get(url, timeout=15, retries=2)
    if not r or r.status_code != 200:
        return None, None
    try:
        m = (r.json() or {}).get("market", {}) or {}
        if m.get("yes_ask_dollars") is not None:
            return float(m["yes_ask_dollars"]), "yes_ask_dollars"
        if m.get("last_price_dollars") is not None:
            return float(m["last_price_dollars"]), "last_price_dollars"
        return None, None
    except Exception:
        return None, None

def read_market_price():
    p, used = fetch_kalshi_yes_price(KALSHI_MARKET_TICKER)
    if p is not None:
        return p, "kalshi", {"ticker": KALSHI_MARKET_TICKER, "field": used}

    # legacy local_json
    try:
        with open("/app/market.json", "r", encoding="utf-8") as f:
            j = json.load(f)
        return float(j["market_price_yes_over_48"]), "local_json", {"path": "/app/market.json"}
    except Exception:
        return float(os.getenv("MARKET_PRICE_YES_OVER_48", "0.17")), "env_fallback", {"env": "MARKET_PRICE_YES_OVER_48"}

# ================= MODELS =================

@dataclass
class SourceSummary:
    src: str
    rows_inserted: int
    status: str
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

    sources: list[SourceSummary]
    removed_outliers: list[str]
    notes: list[str]

    consensus_now_f: float | None
    consensus_last8h_min_f: float | None
    consensus_last8h_max_f: float | None

    consensus_tmr_min_f: float | None
    consensus_tmr_max_f: float | None

    threshold_f: float
    p_over: float | None
    market_price: float
    market_source: str
    market_meta: dict
    edge: float | None
    stake: float | None

# ================= RUN =================

def run_once_struct(q=None, lat=None, lon=None) -> RunResult:
    lat0, lon0, loc_name, tz_guess = resolve_location(q, lat, lon)
    local_tz, tz_name = safe_zoneinfo(tz_guess)

    con = db()
    notes = []

    n1 = insert_rows(con, "open_meteo", lat0, lon0, fetch_open_meteo(lat0, lon0))
    n2 = insert_rows(con, "weather_gov", lat0, lon0, fetch_weather_gov(lat0, lon0))
    n3 = insert_rows(con, "tomorrow_io", lat0, lon0, fetch_tomorrow_io(lat0, lon0))

    by_source = load_recent(con, lat0, lon0, hours=6)

    per_now, per_min8, per_max8 = {}, {}, {}
    per_tmr_min, per_tmr_max = {}, {}

    summaries = []
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

        if now_f is not None: per_now[src] = now_f
        if mn8 is not None: per_min8[src] = mn8
        if mx8 is not None: per_max8[src] = mx8
        if tmin is not None: per_tmr_min[src] = tmin
        if tmax is not None: per_tmr_max[src] = tmax

        inserted = {"open_meteo": n1, "weather_gov": n2, "tomorrow_io": n3}.get(src, 0)

        summaries.append(SourceSummary(
            src=src,
            rows_inserted=inserted,
            status=status,
            now_f=now_f,
            last8h_min_f=mn8,
            last8h_max_f=mx8,
            tmr_min_f=tmin,
            tmr_max_f=tmax,
        ))

    removed = []

    def consensus(vals_dict):
        if not vals_dict:
            return None, []
        filtered = remove_outliers_iqr(vals_dict)
        rem = sorted(set(vals_dict.keys()) - set(filtered.keys()))
        return mean(filtered.values()), rem

    c_now, r1 = consensus(per_now)
    c_min8, r2 = consensus(per_min8)
    c_max8, r3 = consensus(per_max8)
    c_tmr_min, r4 = consensus(per_tmr_min)
    c_tmr_max, r5 = consensus(per_tmr_max)
    removed = sorted(set(r1 + r2 + r3 + r4 + r5))

    market_price, market_source, market_meta = read_market_price()

    threshold = 48.0
    p_over = edge = stake = None
    if c_tmr_max is not None:
        spread = stdev(per_tmr_max.values()) if len(per_tmr_max) > 1 else 2.0
        sigma_total = (2.0**2 + spread**2) ** 0.5
        p_over = monte_carlo_prob_over(threshold, c_tmr_max, sigma_total)
        edge = p_over - market_price
        stake = fractional_kelly(p_over, market_price, fraction=0.25)
    else:
        notes.append("Not enough tomorrow MAX to compute probability.")

    local_tomorrow = (datetime.now(local_tz) + timedelta(days=1)).date().isoformat()
    notes.append(f"Inserted rows: open_meteo={n1}, weather_gov={n2}, tomorrow_io={n3}")

    return RunResult(
        ok=True,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        location_name=loc_name,
        timezone_name=tz_name,
        lat=float(lat0),
        lon=float(lon0),
        local_tomorrow_date=local_tomorrow,
        sources=summaries,
        removed_outliers=removed,
        notes=notes,
        consensus_now_f=c_now,
        consensus_last8h_min_f=c_min8,
        consensus_last8h_max_f=c_max8,
        consensus_tmr_min_f=c_tmr_min,
        consensus_tmr_max_f=c_tmr_max,
        threshold_f=threshold,
        p_over=p_over,
        market_price=market_price,
        market_source=market_source,
        market_meta=market_meta,
        edge=edge,
        stake=stake,
    )

def run_once_text(q=None, lat=None, lon=None) -> str:
    d = run_once_struct(q=q, lat=lat, lon=lon)

    def ff(x): return "N/A" if x is None else f"{x:.1f}F"

    lines = []
    lines.append(f"Generated: {d.generated_at_utc}")
    lines.append(f"Location: {d.location_name} ({d.lat:.4f},{d.lon:.4f}) TZ={d.timezone_name}")
    lines.append(f"Tomorrow (local): {d.local_tomorrow_date}")
    lines.append(f"Market: {d.market_source} price={d.market_price:.4f}")

    lines.append("\nNOW + last 8h consensus:")
    lines.append(f"NOW / MIN / MAX: {ff(d.consensus_now_f)} / {ff(d.consensus_last8h_min_f)} / {ff(d.consensus_last8h_max_f)}")

    lines.append("\nTomorrow consensus MIN/MAX:")
    lines.append(f"MIN / MAX: {ff(d.consensus_tmr_min_f)} / {ff(d.consensus_tmr_max_f)}")

    for s in d.sources:
        lines.append(f"\n[{s.src}] status={s.status} inserted={s.rows_inserted}")
        lines.append(f"Now / 8h MIN / 8h MAX: {ff(s.now_f)} / {ff(s.last8h_min_f)} / {ff(s.last8h_max_f)}")
        lines.append(f"Tomorrow MIN/MAX: {ff(s.tmr_min_f)} / {ff(s.tmr_max_f)}")

    if d.p_over is not None:
        lines.append(f"\nP(TMAX > {d.threshold_f:.0f}F): {d.p_over*100:.2f}%")
        lines.append(f"Edge: {(d.edge*100):.2f}% | Stake(25% Kelly): {(d.stake*100):.2f}%")

    if d.removed_outliers:
        lines.append(f"\nOutliers removed: {d.removed_outliers}")

    if d.notes:
        lines.append("\nNotes:")
        lines.extend([f"- {x}" for x in d.notes])

    return "\n".join(lines)
