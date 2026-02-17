import os
import time
from datetime import datetime, timezone

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse

from app.main import run_once_struct, run_once_text, geocode_us

app = FastAPI()

REFRESH_MINUTES = int(os.getenv("REFRESH_MINUTES", "30"))
CACHE_TTL_SECONDS = max(60, REFRESH_MINUTES * 60)

CACHE = {}  # key -> {"ts": epoch, "data": RunResult|None, "err": str|None}

def fmt_f(x):
    return "N/A" if x is None else f"{x:.1f}°F"

def pct(x):
    return "N/A" if x is None else f"{x*100:.2f}%"

def badge_for_status(status):
    if status == "OK":
        return ("OK", "#22c55e")
    if status == "NO_KEY":
        return ("NO_KEY", "#f97316")
    return ("EMPTY", "#94a3b8")

def cache_key(q, lat, lon):
    if lat is not None and lon is not None:
        return f"latlon:{float(lat):.5f},{float(lon):.5f}"
    q = (q or "").strip().lower()
    return f"q:{q}" if q else "default"

def get_cached(q=None, lat=None, lon=None):
    key = cache_key(q, lat, lon)
    now = time.time()
    it = CACHE.get(key)
    if it and (now - it["ts"] < CACHE_TTL_SECONDS) and it.get("data") is not None:
        return it["data"], key, True, None

    try:
        data = run_once_struct(q=q, lat=lat, lon=lon)
        CACHE[key] = {"ts": now, "data": data, "err": None}
        return data, key, False, None
    except Exception as e:
        CACHE[key] = {"ts": now, "data": None, "err": str(e)}
        return None, key, False, str(e)

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
    try:
        return run_once_text(q=q, lat=lat, lon=lon) + "\n"
    except Exception as e:
        # tránh 500
        return PlainTextResponse(f"ERROR: {e}\n", status_code=200)

@app.get("/api/latest.json")
def api_latest(q: str = None, lat: float = None, lon: float = None):
    d, key, hit, err = get_cached(q=q, lat=lat, lon=lon)
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

@app.get("/", response_class=HTMLResponse)
def home(q: str = None, lat: float = None, lon: float = None):
    d, key, hit, err = get_cached(q=q, lat=lat, lon=lon)
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

    # Build rows
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

          <div class="card">
            <h2 style="margin:0 0 8px;color:var(--muted);font-size:16px;">Tomorrow (Local) MIN / MAX</h2>
            <div class="sub">Consensus MIN/MAX: <strong>{fmt_f(d.consensus_tmr_min_f)} / {fmt_f(d.consensus_tmr_max_f)}</strong></div>
            <table>
              <thead><tr><th class="th">Source</th><th class="th">Status</th><th class="th">Inserted</th><th class="th">Tomorrow MIN</th><th class="th">Tomorrow MAX</th></tr></thead>
              <tbody>{rows_tmr}</tbody>
            </table>
          </div>
        </div>
      </body>
    </html>
    """
