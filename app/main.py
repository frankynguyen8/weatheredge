import os
import json
import time
import sqlite3
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

# ================= OPTIONAL PANDAS =================
try:
    import pandas as pd
except Exception:
    pd = None

# ================= CONFIG =================

DB_PATH = os.getenv("DB_PATH", "/var/data/weatheredge.sqlite")

DEFAULT_LAT = float(os.getenv("LAT", "40.78858"))
DEFAULT_LON = float(os.getenv("LON", "-73.9661"))

TOMORROW_KEY = os.getenv("TOMORROW_KEY", "").strip()

EXPECTED_SOURCES = ["open_meteo", "weather_gov", "tomorrow_io"]

# Kalshi read-only
KALSHI_ACCESS_KEY = os.getenv("KALSHI_ACCESS_KEY", "").strip()
KALSHI_BASE_URL = os.getenv("KALSHI_BASE_URL", "https://api.elections.kalshi.com").rstrip("/")
KALSHI_MARKET_TICKER = os.getenv("KALSHI_MARKET_TICKER", "").strip()


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
    import random, math
    random.seed(seed)
    sigma = max(0.1, float(sigma_f))
    count = 0
    for _ in range(sims):
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


# ================= WEATHER FETCHERS =================

def fetch_open_meteo(lat, lon, horizon_hours=48):
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m"
        "&temperature_unit=fahrenheit"
        "&timezone=UTC"
    )
    r = safe_get(url, timeout=20, retries=3)
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
    p = safe_get(f"https://api.weather.gov/points/{lat},{lon}", headers=headers, timeout=20, retries=3)
    if not p or p.status_code != 200:
        return []
    try:
        forecast_url = p.json()["properties"]["forecastHourly"]
    except Exception:
        return []

    r = safe_get(forecast_url, headers=headers, timeout=20, retries=3)
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


def fetch_tomorrow_io_timelines(lat, lon, horizon_hours=48):
    """
    Tomorrow.io Timelines API (POST /v4/timelines) - lấy forecast theo giờ.
    """
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
    r = safe_post(url, headers=headers, json_body=payload, timeout=20, retries=3)
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


def fetch_tomorrow_io_forecast_daily(lat, lon, location_text=None, units="imperial", timesteps=("1d",)):
    """
    Tomorrow.io Weather Forecast API (GET /v4/weather/forecast)
    - Mẫu gốc bạn đưa là JS dùng querystring: apikey, location, units, timesteps. :contentReference[oaicite:3]{index=3}
    - Ở đây dùng requests.get.
    """
    if not TOMORROW_KEY:
        return None, "Missing TOMORROW_KEY"

    base_url = "https://api.tomorrow.io/v4/weather/forecast"
    loc = location_text.strip() if location_text else f"{lat},{lon}"

    params = {
        "apikey": TOMORROW_KEY,
        "location": loc,
        "units": units,
        "timesteps": ",".join(list(timesteps)),
    }
    headers = {"accept": "application/json"}

    r = safe_get(base_url, headers=headers, timeout=20, retries=3)
    # safe_get không nhận params, nên build URL thủ công:
    if r is None:
        # thử lại theo cách đúng với params
        try:
            r = requests.get(base_url, params=params, headers=headers, timeout=20)
        except Exception as e:
            return None, str(e)
    else:
        # Nếu safe_get gọi base_url không có params thì sai; ta gọi lại đúng:
        try:
            r = requests.get(base_url, params=params, headers=headers, timeout=20)
        except Exception as e:
            return None, str(e)

    if r.status_code != 200:
        return None, f"HTTP {r.status_code}: {r.text[:300]}"

    try:
        data = r.json()
        daily = (data.get("timelines") or {}).get("daily") or []
        # Trả về list dict: time + temperatureMin/Max/Avg nếu có
        out = []
        for d in daily:
            values = d.get("values") or {}
            out.append({
                "time": d.get("time"),
                "temperatureMax": values.get("temperatureMax"),
                "temperatureMin": values.get("temperatureMin"),
                "temperatureAvg": values.get("temperatureAvg"),
            })
        return out, None
    except Exception as e:
        return None, str(e)


def fetch_tomorrow_io_historical_df(
    location,
    fields=("temperature", "humidity"),
    timesteps=("1h",),
    start_time="nowMinus30d",
    end_time="nowMinus15d",
):
    """
    Tomorrow.io Historical API (POST /v4/historical) + Pandas DataFrame. :contentReference[oaicite:4]{index=4}
    """
    if not TOMORROW_KEY:
        return None, "Missing TOMORROW_KEY"
    if pd is None:
        return None, "pandas is not installed. Install: pip install pandas"

    url = f"https://api.tomorrow.io/v4/historical?apikey={TOMORROW_KEY}"
    payload = {
        "location": location,
        "fields": list(fields),
        "timesteps": list(timesteps),
        "startTime": start_time,
        "endTime": end_time,
    }
    headers = {
        "accept": "application/json",
        "Accept-Encoding": "gzip",
        "content-type": "application/json",
    }

    r = safe_post(url, headers=headers, json_body=payload, timeout=30, retries=3)
    if not r:
        return None, "No response"
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}: {r.text[:300]}"

    try:
        j = r.json()
        timelines = j["data"]["timelines"]
        if not timelines:
            return pd.DataFrame(), None
        timeline = timelines[0]
        df = pd.DataFrame(timeline["intervals"])
        # flatten values
        df = pd.concat([df.drop(["values"], axis=1), df["values"].apply(pd.Series)], axis=1)
        return df, None
    except Exception as e:
        return None, str(e)


def summarize_historical_df(df):
    """
    Một phân tích đơn giản: stats theo cột numeric.
    """
    if pd is None:
        return None
    if df is None or df.empty:
        return {"rows": 0, "summary": {}}
    numeric_cols = [c for c in df.columns if c not in ("time",) and pd.api.types.is_numeric_dtype(df[c])]
    summary = {}
    for c in numeric_cols:
        s = df[c].dropna()
        if len(s) == 0:
            continue
        summary[c] = {
            "count": int(s.count()),
            "mean": float(s.mean()),
            "min": float(s.min()),
            "max": float(s.max()),
            "p50": float(s.quantile(0.50)),
            "p90": float(s.quantile(0.90)),
        }
    return {"rows": int(len(df)), "summary": summary}


# ================= KALSHI READ-ONLY (NO RSA) =================

def kalshi_headers_readonly():
    h = {"Accept": "application/json"}
    if KALSHI_ACCESS_KEY:
        h["KALSHI-ACCESS-KEY"] = KALSHI_ACCESS_KEY
        # if your environment requires bearer, enable:
        # h["Authorization"] = f"Bearer {KALSHI_ACCESS_KEY}"
    return h


def kalshi_get_readonly(path: str):
    url = KALSHI_BASE_URL + path
    r = safe_get(url, headers=kalshi_headers_readonly(), timeout=20, retries=2)
    if not r:
        return None, "No response"
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}: {r.text[:200]}"
    try:
        return r.json(), None
    except Exception as e:
        return None, str(e)


def kalshi_get_market(ticker: str):
    if not ticker:
        return None, "Missing ticker"
    return kalshi_get_readonly(f"/trade-api/v2/markets/{ticker}")


def extract_yes_price_dollars(market_obj: dict):
    if not market_obj:
        return None, None
    m = market_obj.get("market") if "market" in market_obj else market_obj

    for field in ("yes_ask_dollars", "yes_bid_dollars", "last_price_dollars"):
        v = m.get(field)
        if v is not None:
            try:
                return float(v), field
            except Exception:
                pass

    for field in ("yes_ask", "yes_bid", "last_price"):
        v = m.get(field)
        if v is not None:
            try:
                return float(v) / 100.0, field
            except Exception:
                pass

    return None, None


def fetch_kalshi_yes_price(ticker: str):
    data, err = kalshi_get_market(ticker)
    if err:
        return None, None, err
    price, field = extract_yes_price_dollars(data)
    if price is None:
        return None, None, "No YES price fields in response"
    return price, field, None


def read_market_price():
    if KALSHI_MARKET_TICKER:
        p, field, err = fetch_kalshi_yes_price(KALSHI_MARKET_TICKER)
        if p is not None:
            return p, "kalshi", {"ticker": KALSHI_MARKET_TICKER, "field": field}
        fallback = float(os.getenv("MARKET_PRICE_YES_OVER_48", "0.17"))
        return fallback, "env_fallback", {"env": "MARKET_PRICE_YES_OVER_48", "kalshi_error": err}

    fallback = float(os.getenv("MARKET_PRICE_YES_OVER_48", "0.17"))
    return fallback, "env_fallback", {"env": "MARKET_PRICE_YES_OVER_48", "kalshi_error": "Missing KALSHI_MARKET_TICKER"}


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

    # added: forecast daily + historical analysis
    tomorrow_forecast_daily: list[dict] | None
    historical_summary: dict | None
    historical_csv_path: str | None


# ================= RUN =================

def run_once_struct(
    q=None,
    lat=None,
    lon=None,
    threshold_f=48.0,
    include_forecast_daily=True,
    include_historical=False,
    historical_start="nowMinus30d",
    historical_end="nowMinus15d",
    historical_fields=("temperature", "humidity"),
    historical_timesteps=("1h",),
    historical_csv_dir=".",
) -> RunResult:
    lat0, lon0, loc_name, tz_guess = resolve_location(q, lat, lon)
    local_tz, tz_name = safe_zoneinfo(tz_guess)

    con = db()
    notes = []

    # hourly forecast sources
    n1 = insert_rows(con, "open_meteo", lat0, lon0, fetch_open_meteo(lat0, lon0))
    n2 = insert_rows(con, "weather_gov", lat0, lon0, fetch_weather_gov(lat0, lon0))
    n3 = insert_rows(con, "tomorrow_io", lat0, lon0, fetch_tomorrow_io_timelines(lat0, lon0))

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

        if now_f is not None:
            per_now[src] = now_f
        if mn8 is not None:
            per_min8[src] = mn8
        if mx8 is not None:
            per_max8[src] = mx8
        if tmin is not None:
            per_tmr_min[src] = tmin
        if tmax is not None:
            per_tmr_max[src] = tmax

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

    p_over = edge = stake = None
    if c_tmr_max is not None:
        spread = stdev(per_tmr_max.values()) if len(per_tmr_max) > 1 else 2.0
        sigma_total = (2.0**2 + spread**2) ** 0.5
        p_over = monte_carlo_prob_over(float(threshold_f), c_tmr_max, sigma_total)
        edge = p_over - market_price
        stake = fractional_kelly(p_over, market_price, fraction=0.25)
    else:
        notes.append("Not enough tomorrow MAX to compute probability.")

    local_tomorrow = (datetime.now(local_tz) + timedelta(days=1)).date().isoformat()
    notes.append(f"Inserted rows: open_meteo={n1}, weather_gov={n2}, tomorrow_io={n3}")

    # Added: Tomorrow Forecast Daily (GET /v4/weather/forecast)
    forecast_daily = None
    if include_forecast_daily:
        forecast_daily, err = fetch_tomorrow_io_forecast_daily(lat0, lon0, location_text=None, units="imperial", timesteps=("1d",))
        if err:
            notes.append(f"Tomorrow.io forecast daily error: {err}")

    # Added: Historical + Pandas
    hist_summary = None
    hist_csv = None
    if include_historical:
        location_str = f"{lat0},{lon0}"
        df, err = fetch_tomorrow_io_historical_df(
            location=location_str,
            fields=historical_fields,
            timesteps=historical_timesteps,
            start_time=historical_start,
            end_time=historical_end,
        )
        if err:
            notes.append(f"Historical fetch error: {err}")
        else:
            hist_summary = summarize_historical_df(df)
            # save CSV
            if df is not None and pd is not None:
                os.makedirs(historical_csv_dir, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                hist_csv = os.path.join(historical_csv_dir, f"historical_{location_str.replace(',', '_')}_{ts}.csv")
                df.to_csv(hist_csv, index=False)
                notes.append(f"Saved historical CSV: {hist_csv}")

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
        threshold_f=float(threshold_f),
        p_over=p_over,
        market_price=market_price,
        market_source=market_source,
        market_meta=market_meta,
        edge=edge,
        stake=stake,
        tomorrow_forecast_daily=forecast_daily,
        historical_summary=hist_summary,
        historical_csv_path=hist_csv,
    )


def run_once_text(**kwargs) -> str:
    d = run_once_struct(**kwargs)

    def ff(x):
        return "N/A" if x is None else f"{x:.1f}F"

    lines = []
    lines.append(f"Generated: {d.generated_at_utc}")
    lines.append(f"Location: {d.location_name} ({d.lat:.4f},{d.lon:.4f}) TZ={d.timezone_name}")
    lines.append(f"Tomorrow (local): {d.local_tomorrow_date}")
    lines.append(f"Market: {d.market_source} price={d.market_price:.4f} meta={d.market_meta}")
    lines.append(f"Consensus NOW/MIN/MAX(8h): {ff(d.consensus_now_f)} / {ff(d.consensus_last8h_min_f)} / {ff(d.consensus_last8h_max_f)}")
    lines.append(f"Consensus TMR MIN/MAX: {ff(d.consensus_tmr_min_f)} / {ff(d.consensus_tmr_max_f)}")
    if d.p_over is not None:
        lines.append(f"P(temp > {d.threshold_f:.1f}F) ≈ {d.p_over:.4f} | edge ≈ {d.edge:.4f} | stake(kelly_frac) ≈ {d.stake:.4f}")
    if d.tomorrow_forecast_daily is not None:
        lines.append("Tomorrow.io Forecast Daily (GET /v4/weather/forecast):")
        lines.append(json.dumps(d.tomorrow_forecast_daily[:5], indent=2) if d.tomorrow_forecast_daily else "[]")
    if d.historical_summary is not None:
        lines.append("Historical summary (from Tomorrow.io /v4/historical):")
        lines.append(json.dumps(d.historical_summary, indent=2))
    if d.historical_csv_path:
        lines.append(f"Historical CSV saved at: {d.historical_csv_path}")
    if d.removed_outliers:
        lines.append(f"Removed outliers (IQR): {d.removed_outliers}")
    if d.notes:
        lines.append("Notes:")
        lines.extend([f"- {x}" for x in d.notes])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="WeatherEdge: Forecast + Tomorrow.io forecast/historical integration")
    parser.add_argument("--q", type=str, default=None, help="Location query (US), e.g. 'New York, NY'")
    parser.add_argument("--lat", type=float, default=None, help="Latitude")
    parser.add_argument("--lon", type=float, default=None, help="Longitude")
    parser.add_argument("--threshold", type=float, default=48.0, help="Threshold temperature in F for probability calc")

    parser.add_argument("--no-forecast-daily", action="store_true", help="Disable Tomorrow.io /v4/weather/forecast call")
    parser.add_argument("--historical", action="store_true", help="Enable Tomorrow.io /v4/historical + pandas analysis")
    parser.add_argument("--hist-start", type=str, default="nowMinus30d", help="Historical startTime")
    parser.add_argument("--hist-end", type=str, default="nowMinus15d", help="Historical endTime")
    parser.add_argument("--hist-fields", type=str, default="temperature,humidity", help="Comma-separated fields")
    parser.add_argument("--hist-timesteps", type=str, default="1h", help="Comma-separated timesteps")
    parser.add_argument("--hist-csv-dir", type=str, default=".", help="Where to save historical CSV")

    args = parser.parse_args()

    text = run_once_text(
        q=args.q,
        lat=args.lat,
        lon=args.lon,
        threshold_f=args.threshold,
        include_forecast_daily=(not args.no_forecast_daily),
        include_historical=args.historical,
        historical_start=args.hist_start,
        historical_end=args.hist_end,
        historical_fields=tuple([x.strip() for x in args.hist_fields.split(",") if x.strip()]),
        historical_timesteps=tuple([x.strip() for x in args.hist_timesteps.split(",") if x.strip()]),
        historical_csv_dir=args.hist_csv_dir,
    )
    print(text)


if __name__ == "__main__":
    main()
