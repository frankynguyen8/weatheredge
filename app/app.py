import os
import time
from datetime import datetime, timezone

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, PlainTextResponse

# NOTE:
# - app.py của bạn đang import từ app.main. :contentReference[oaicite:1]{index=1}
# - Hãy đảm bảo app/main.py có các hàm này (từ weatheredge_full.py):
#   run_once_struct, run_once_text, geocode_us
from app.main import run_once_struct, run_once_text, geocode_us

import requests

app = FastAPI()

REFRESH_MINUTES = int(os.getenv("REFRESH_MINUTES", "30"))
CACHE_TTL_SECONDS = max(60, REFRESH_MINUTES * 60)

CACHE = {}  # key -> {"ts": epoch, "data": RunResult|None, "err": str|None}


# ================== UI helpers ==================

def fmt_f(x):
    return "N/A" if x is None else f"{x:.1f}°F"

def pct(x):
    return "N/A" if x is None else f"{x * 100:.2f}%"

def badge_for_status(status):
    if status == "OK":
        return ("OK", "#22c55e")
    if status == "NO_KEY":
        return ("NO_KEY", "#f97316")
    return ("EMPTY", "#94a3b8")

def cache_key(q, lat, lon, extra=""):
    """
    Cache key cho run default (không historical).
    Nếu historical bật, tách key riêng để tránh cache nhầm.
    """
    if lat is not None and lon is not None:
        base = f"latlon:{float(lat):.5f},{float(lon):.5f}"
    else:
        qq = (q or "").strip().lower()
        base = f"q:{qq}" if qq else "default"
    return base + (f"|{extra}" if extra else "")


# ================== Kalshi series search (self-contained) ==================
# app.py cũ có /api/kalshi/search và import kalshi_search_series :contentReference[oaicite:2]{index=2}
# Mình đưa luôn vào đây để không còn ImportError.

KALSHI_ACCESS_KEY = os.getenv("KALSHI_ACCESS_KEY", "").strip()
KALSHI_BASE_URL = os.getenv("KALSHI_BASE_URL", "https://api.elections.kalshi.com").rstrip("/")

def _kalshi_headers():
    h = {"Accept": "application/json"}
    if KALSHI_ACCESS_KEY:
        # Một số môi trường dùng header này
        h["KALSHI-ACCESS-KEY"] = KALSHI_ACCESS_KEY
        # Nếu account bạn require bearer, bật dòng dưới:
        # h["Authorization"] = f"Bearer {KALSHI_ACCESS_KEY}"
    return h

def kalshi_search_series(q: str, limit: int = 10):
    """
    Tìm series trên Kalshi.
    Endpoint có thể khác nhau tùy môi trường; mình thử 2 kiểu phổ biến:
    - /trade-api/v2/series?search=...&limit=...
    - /trade-api/v2/series?query=...&limit=...
    """
    q = (q or "").strip()
    if not q:
        return None, "Missing query"

    candidates = [
        f"{KALSHI_BASE_URL}/trade-api/v2/series?search={requests.utils.quote(q)}&limit={int(limit)}",
        f"{KALSHI_BASE_URL}/trade-api/v2/series?query={requests.utils.quote(q)}&limit={int(limit)}",
    ]

    last_err = None
    for url in candidates:
        try:
            r = requests.get(url, headers=_kalshi_headers(), timeout=20)
            if r.status_code != 200:
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                continue
            data = r.json()
            # vài schema hay gặp: {"series":[...]} hoặc {"data":[...]}
            series = data.get("series") or data.get("data") or data.get("results")
            if series is None:
                # nếu response là list
                if isinstance(data, list):
                    series = data
            if series is None:
                last_err = "Unexpected response schema"
                continue
            return series, None
        except Exception as e:
            last_err = str(e)

    return None, last_err or "Kalshi search failed"


# ================== Caching ==================

def get_cached_run(q=None, lat=None, lon=None):
    """
    Cache cho run mặc định (không bật historical).
    """
    key = cache_key(q, lat, lon, extra="default")
    now = time.time()
    it = CACHE.get(key)
    if it and (now - it["ts"] < CACHE_TTL_SECONDS) and it.get("data") is not None:
        return it["data"], key, True, None

    try:
        data = run_once_struct(
            q=q, lat=lat, lon=lon,
            include_forecast_daily=True,
            include_historical=False,
        )
        CACHE[key] = {"ts": now, "data": data, "err": None}
        return data, key, False, None
    except Exception as e:
        CACHE[key] = {"ts": now, "data": None, "err": str(e)}
        return None, key, False, str(e)


def run_historical(q=None, lat=None, lon=None,
                   hist_start="nowMinus30d",
                   hist_end="nowMinus15d",
                   hist_fields="temperature,humidity",
                   hist_timesteps="1h",
                   hist_csv_dir="."):
    """
    Không cache historical theo TTL mặc định (vì nặng) – nhưng bạn có thể bật cache nếu muốn.
    """
    fields = tuple([x.strip() for x in (hist_fields or "").split(",") if x.strip()])
    timesteps = tuple([x.strip() for x in (hist_timesteps or "").split(",") if x.strip()])

    return run_once_struct(
        q=q, lat=lat, lon=lon,
        include_forecast_daily=True,
        include_historical=True,
        historical_start=hist_start,
        historical_end=hist_end,
        historical_fields=fields,
        historical_timesteps=timesteps,
        historical_csv_dir=hist_csv_dir,
    )


# ================== Routes ==================

@app.get("/health")
def health():
    return {
        "ok": True,
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "cache_keys_sample": list(CACHE.keys())[:20],
        "server_time_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/geocode")
def api_geocode(q: str = Query(..., min_length=2)):
    return {"ok": True, "results": geocode_us(q, count=7)}


@app.get("/api/run", response_class=PlainTextResponse)
def api_run(q: str = None, lat: float = None, lon: float = None):
    """
    Text output kiểu cũ.
    """
    try:
        return run_once_text(q=q, lat=lat, lon=lon) + "\n"
    except Exception as e:
        return PlainTextResponse(f"ERROR: {e}\n", status_code=200)


@app.get("/api/latest.json")
def api_latest(
    q: str = None,
    lat: float = None,
    lon: float = None,
):
    """
    Latest cached run (không historical), nhưng CÓ forecast daily.
    """
    d, key, hit, err = get_cached_run(q=q, lat=lat, lon=lon)
    if err:
        return {"ok": False, "cache_key": key, "cache_hit": hit, "error": err}

    return {
        "ok": True,
        "cache_key": key,
        "cache_hit": hit,
        "cache_ttl_seconds": CACHE_TTL_SECONDS,

        "generated_at_utc": d.generated_at_utc,
        "location_name": d.location_name,
        "timezone_name": d.timezone_name,
        "lat": d.lat,
        "lon": d.lon,
        "local_tomorrow_date": d.local_tomorrow_date,

        "consensus_now_f": d.consensus_now_f,
        "consensus_last8h_min_f": d.consensus_last8h_min_f,
        "consensus_last8h_max_f": d.consensus_last8h_max_f,

        "consensus_tmr_min_f": d.consensus_tmr_min_f,
        "consensus_tmr_max_f": d.consensus_tmr_max_f,

        "threshold_f": d.threshold_f,
        "p_over": d.p_over,
        "market_price": d.market_price,
        "market_source": d.market_source,
        "market_meta": d.market_meta,
        "edge": d.edge,
        "stake": d.stake,

        "removed_outliers": d.removed_outliers,
        "notes": d.notes,

        # NEW: Tomorrow.io forecast daily (GET /v4/weather/forecast)
        "tomorrow_forecast_daily": d.tomorrow_forecast_daily,

        # NEW: Historical (disabled here)
        "historical_summary": d.historical_summary,
        "historical_csv_path": d.historical_csv_path,

        "sources": [
            {
                "src": s.src,
                "status": s.status,
                "rows_inserted": s.rows_inserted,
                "now_f": s.now_f,
                "last8h_min_f": s.last8h_min_f,
                "last8h_max_f": s.last8h_max_f,
                "tomorrow_min_f": s.tmr_min_f,
                "tomorrow_max_f": s.tmr_max_f,
            } for s in d.sources
        ],
    }


@app.get("/api/historical.json")
def api_historical(
    q: str = None,
    lat: float = None,
    lon: float = None,
    hist_start: str = "nowMinus30d",
    hist_end: str = "nowMinus15d",
    hist_fields: str = "temperature,humidity",
    hist_timesteps: str = "1h",
    hist_csv_dir: str = ".",
):
    """
    Bật historical (Tomorrow.io /v4/historical + Pandas analysis).
    """
    try:
        d = run_historical(
            q=q, lat=lat, lon=lon,
            hist_start=hist_start,
            hist_end=hist_end,
            hist_fields=hist_fields,
            hist_timesteps=hist_timesteps,
            hist_csv_dir=hist_csv_dir,
        )
        return {
            "ok": True,
            "generated_at_utc": d.generated_at_utc,
            "location_name": d.location_name,
            "timezone_name": d.timezone_name,
            "lat": d.lat,
            "lon": d.lon,
            "historical_summary": d.historical_summary,
            "historical_csv_path": d.historical_csv_path,
            "notes": d.notes,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/api/kalshi/search")
def api_kalshi_search(q: str = Query(..., min_length=2), limit: int = 10):
    series, err = kalshi_search_series(q, limit=limit)
    if err:
        return {"ok": False, "error": err}

    out = []
    for s in series:
        if not isinstance(s, dict):
            continue
        out.append({
            "ticker": s.get("ticker") or s.get("series_ticker") or s.get("id"),
            "title": s.get("title") or s.get("name") or s.get("question") or s.get("subtitle"),
        })
    return {"ok": True, "results": out}


@app.get("/", response_class=HTMLResponse)
def home(q: str = None, lat: float = None, lon: float = None):
    """
    Trang HTML: giữ layout cũ, bổ sung 2 card:
    - Tomorrow.io Forecast Daily
    - Link Historical
    """
    d, key, hit, err = get_cached_run(q=q, lat=lat, lon=lon)
    q_val = (q or "").replace('"', "&quot;")

    if err:
        return f"""
        <html><head><meta name="viewport" content="width=device-width, initial-scale=1"/></head>
        <body style="font-family:ui-sans-serif,system-ui;padding:18px;background:#0b1020;color:#eaeefc;">
          <h2 style="margin:0 0 10px;">WeatherEdge (USA)</h2>
          <p style="opacity:.85;margin:0 0 10px;">Cache key: <code>{key}</code> · hit: <code>{str(hit).lower()}</code></p>
          <div style="margin:10px 0; padding:12px; border-radius:14px; background:#121a33;">
            <div style="opacity:.75;margin-bottom:6px;">Error</div>
            <pre style="white-space:pre-wrap;margin:0;">{err}</pre>
          </div>
          <form method="get" action="/" style="display:flex;gap:10px;flex-wrap:wrap;">
            <input name="q" value="{q_val}" placeholder='City, State (ex: "Los Angeles, CA")'
              style="flex:1;min-width:220px;padding:10px 12px;border-radius:12px;border:1px solid rgba(255,255,255,.2);background:#0b1020;color:#eaeefc;"/>
            <button type="submit" style="padding:10px 14px;border-radius:12px;border:1px solid rgba(255,255,255,.2);background:#223056;color:#eaeefc;cursor:pointer;">Search</button>
          </form>
          <p style="margin-top:12px;">
            <a style="color:#7aa2ff" href="/api/latest.json">/api/latest.json</a> ·
            <a style="color:#7aa2ff" href="/api/run">/api/run</a> ·
            <a style="color:#7aa2ff" href="/health">/health</a>
          </p>
        </body></html>
        """

    rows_now = ""
    for s in d.sources:
        label, color = badge_for_status(s.status)
        rows_now += f"""
        <tr>
          <td class="td mono">{s.src}</td>
          <td class="td"><span class="tag" style="border-color:{color};color:{color};">{label}</span></td>
          <td class="td">{fmt_f(s.now_f)}</td>
          <td class="td">{fmt_f(s.last8h_min_f)}</td>
          <td class="td">{fmt_f(s.last8h_max_f)}</td>
        </tr>
        """

    rows_tmr = ""
    for s in d.sources:
        label, color = badge_for_status(s.status)
        rows_tmr += f"""
        <tr>
          <td class="td mono">{s.src}</td>
          <td class="td"><span class="tag" style="border-color:{color};color:{color};">{label}</span></td>
          <td class="td mono">{s.rows_inserted}</td>
          <td class="td">{fmt_f(s.tmr_min_f)}</td>
          <td class="td">{fmt_f(s.tmr_max_f)}</td>
        </tr>
        """

    market_card = f"""
    <div class="card">
      <h2 style="margin:0 0 8px;color:var(--muted);font-size:16px;">Market (Kalshi)</h2>
      <div class="sub">
        Source: <strong>{d.market_source}</strong>
        · YES Price: <strong>{(d.market_price*100):.2f}%</strong>
      </div>
      <div class="sub" style="margin-top:6px;">
        P(TMAX &gt; 48F): <strong>{pct(d.p_over) if d.p_over is not None else "N/A"}</strong>
        · Edge: <strong>{pct(d.edge) if d.edge is not None else "N/A"}</strong>
        · Stake: <strong>{pct(d.stake) if d.stake is not None else "N/A"}</strong>
      </div>
      <div class="sub" style="margin-top:6px;">
        Meta: <span class="mono">{str(d.market_meta)}</span>
      </div>
      <div class="sub" style="margin-top:6px;">
        Find ticker: <span class="mono">/api/kalshi/search?q=KXHIGHNY</span>
      </div>
    </div>
    """

    # NEW: forecast daily card
    forecast_preview = ""
    if d.tomorrow_forecast_daily is None:
        forecast_preview = '<div class="sub">No daily forecast (check TOMORROW_KEY).</div>'
    else:
        # show first 5 days
        items = d.tomorrow_forecast_daily[:5]
        lis = ""
        for it in items:
            t = it.get("time")
            tmax = it.get("temperatureMax")
            tmin = it.get("temperatureMin")
            lis += f"<li class='sub'><span class='mono'>{t}</span> · min={tmin} · max={tmax}</li>"
        forecast_preview = f"<ul style='margin:8px 0 0 18px;padding:0;'>{lis}</ul>"

    forecast_card = f"""
    <div class="card">
      <h2 style="margin:0 0 8px;color:var(--muted);font-size:16px;">Tomorrow.io Forecast Daily</h2>
      <div class="sub">From: <span class="mono">GET /v4/weather/forecast</span></div>
      {forecast_preview}
      <div class="sub" style="margin-top:10px;">Raw JSON: <span class="mono">/api/latest.json</span></div>
    </div>
    """

    historical_card = f"""
    <div class="card">
      <h2 style="margin:0 0 8px;color:var(--muted);font-size:16px;">Historical (Tomorrow.io)</h2>
      <div class="sub">Run historical + pandas analysis:</div>
      <div class="sub"><span class="mono">/api/historical.json?q=Seattle, WA&hist_start=nowMinus30d&hist_end=nowMinus15d</span></div>
      <div class="sub" style="margin-top:6px;">Tip: historical cần <span class="mono">pandas</span> và TOMORROW_KEY.</div>
    </div>
    """

    return f"""
    <html>
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1"/>
        <meta http-equiv="refresh" content="{CACHE_TTL_SECONDS}">
        <title>WeatherEdge USA</title>
        <style>
          :root{{--bg:#0b1020;--text:#eaeefc;--muted:rgba(234,238,252,.72);--border:rgba(255,255,255,.10);--shadow:0 10px 30px rgba(0,0,0,.35);--radius:18px;}}
          *{{box-sizing:border-box;}}
          body{{margin:0;background:radial-gradient(1200px 700px at 20% 0%, rgba(122,162,255,.18), transparent 60%),radial-gradient(1000px 600px at 90% 10%, rgba(34,197,94,.14), transparent 55%),var(--bg);color:var(--text);font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial;}}
          .wrap{{max-width:1040px;margin:0 auto;padding:22px 16px 42px;}}
          .top{{display:flex;align-items:flex-end;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:14px;}}
          h1{{margin:0;font-size:28px;}}
          .sub{{color:var(--muted);font-size:13px;margin-top:6px;}}
          .mono{{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono",monospace;}}
          .card{{background:linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.03));border:1px solid var(--border);border-radius:var(--radius);box-shadow:var(--shadow);padding:14px;margin-top:14px;}}
          .search{{display:flex;gap:10px;flex-wrap:wrap;align-items:center;padding:10px;border-radius:14px;border:1px solid rgba(255,255,255,.10);background:rgba(255,255,255,.04);}}
          .search input{{flex:1;min-width:220px;padding:10px 12px;border-radius:12px;border:1px solid rgba(255,255,255,.14);background:rgba(0,0,0,.20);color:var(--text);outline:none;}}
          .search button{{padding:10px 14px;border-radius:12px;border:1px solid rgba(255,255,255,.14);background:rgba(122,162,255,.16);color:var(--text);cursor:pointer;}}
          table{{width:100%;border-collapse:collapse;border-radius:14px;overflow:hidden;margin-top:10px;}}
          .th{{text-align:left;padding:10px;color:var(--muted);font-size:12px;border-bottom:1px solid rgba(255,255,255,.10);}}
          .td{{padding:10px;border-bottom:1px solid rgba(255,255,255,.06);font-size:13px;}}
          .tag{{display:inline-flex;align-items:center;justify-content:center;padding:4px 10px;border-radius:999px;border:1px solid rgba(255,255,255,.16);background:rgba(255,255,255,.04);font-size:12px;}}
        </style>
      </head>
      <body>
        <div class="wrap">
          <div class="top">
            <div>
              <h1>WeatherEdge (USA)</h1>
              <div class="sub">
                Location: <strong>{d.location_name}</strong> · <span class="mono">{d.lat:.4f},{d.lon:.4f}</span> · TZ: <span class="mono">{d.timezone_name}</span>
              </div>
              <div class="sub">
                Tomorrow (local): <strong>{d.local_tomorrow_date}</strong> · Cache: <span class="mono">{key}</span> · hit: <strong>{str(hit).lower()}</strong>
              </div>
            </div>
            <div class="sub">TTL: <strong>{REFRESH_MINUTES}m</strong></div>
          </div>

          <div class="card">
            <h2 style="margin:0 0 10px;color:var(--muted);font-size:16px;">Search anywhere in USA</h2>
            <form class="search" method="get" action="/">
              <input name="q" value="{q_val}" placeholder='City, State (ex: "Los Angeles, CA") / "Honolulu, HI"' />
              <button type="submit">Search</button>
            </form>
            <div class="sub" style="margin-top:8px;">JSON: <span class="mono">/api/latest.json?q=Seattle, WA</span> · lat/lon: <span class="mono">/?lat=34.0522&lon=-118.2437</span></div>
          </div>

          <div class="card">
            <h2 style="margin:0 0 8px;color:var(--muted);font-size:16px;">NOW + Last 8 Hours</h2>
            <div class="sub">Consensus NOW/MIN/MAX: <strong>{fmt_f(d.consensus_now_f)} / {fmt_f(d.consensus_last8h_min_f)} / {fmt_f(d.consensus_last8h_max_f)}</strong></div>
            <table>
              <thead><tr><th class="th">Source</th><th class="th">Status</th><th class="th">NOW</th><th class="th">8h MIN</th><th class="th">8h MAX</th></tr></thead>
              <tbody>{rows_now}</tbody>
            </table>
          </div>

          {market_card}

          <div class="card">
            <h2 style="margin:0 0 8px;color:var(--muted);font-size:16px;">Tomorrow (Local) MIN / MAX</h2>
            <div class="sub">Consensus MIN/MAX: <strong>{fmt_f(d.consensus_tmr_min_f)} / {fmt_f(d.consensus_tmr_max_f)}</strong></div>
            <table>
              <thead><tr><th class="th">Source</th><th class="th">Status</th><th class="th">Inserted</th><th class="th">Tomorrow MIN</th><th class="th">Tomorrow MAX</th></tr></thead>
              <tbody>{rows_tmr}</tbody>
            </table>
          </div>

          {forecast_card}
          {historical_card}
        </div>
      </body>
    </html>
    """
