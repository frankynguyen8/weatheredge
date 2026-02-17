import os
import time
from datetime import datetime, timezone

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse

from app.main import run_once_struct, run_once_text, geocode_us

app = FastAPI()

REFRESH_MINUTES = int(os.getenv("REFRESH_MINUTES", "30"))
CACHE_TTL_SECONDS = max(60, REFRESH_MINUTES * 60)

# Cache per location key
LOCATION_CACHE = {}  # key -> {"ts": epoch, "data": RunResult}

def fmt_f(x):
    return "N/A" if x is None else f"{x:.1f}°F"

def pct(x):
    return "N/A" if x is None else f"{x*100:.2f}%"

def badge_for_status(status: str):
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
    it = LOCATION_CACHE.get(key)
    if it and (now - it["ts"] < CACHE_TTL_SECONDS):
        return it["data"], key, True

    data = run_once_struct(q=q, lat=lat, lon=lon)
    LOCATION_CACHE[key] = {"ts": now, "data": data}
    return data, key, False

@app.get("/health")
def health():
    return {"ok": True, "cache_ttl_seconds": CACHE_TTL_SECONDS, "cache_keys": list(LOCATION_CACHE.keys())[:20]}

@app.get("/api/run", response_class=PlainTextResponse)
def api_run(q: str | None = None, lat: float | None = None, lon: float | None = None):
    try:
        return run_once_text(q=q, lat=lat, lon=lon) + "\n"
    except Exception as e:
        return PlainTextResponse(f"Service degraded: {e}\n", status_code=200)

@app.get("/api/latest.json")
def api_latest(q: str | None = None, lat: float | None = None, lon: float | None = None):
    try:
        d, key, hit = get_cached(q=q, lat=lat, lon=lon)
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
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=200)

@app.get("/api/geocode")
def api_geocode(q: str = Query(..., min_length=2)):
    # returns US candidates for UI / debug
    return {"ok": True, "results": geocode_us(q, count=7)}

@app.get("/", response_class=HTMLResponse)
def home(q: str | None = None, lat: float | None = None, lon: float | None = None):
    d, key, hit = get_cached(q=q, lat=lat, lon=lon)

    edge = d.edge
    status_color = "#22c55e" if (edge is not None and edge > 0) else "#f97316"
    status_text = "EDGE+" if (edge is not None and edge > 0) else "NO EDGE"

    # tables
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

    notes_html = ""
    if d.notes:
        notes_html = "<ul class='notes'>" + "".join([f"<li>{n}</li>" for n in d.notes]) + "</ul>"

    # form value
    q_val = (q or "").replace('"', "&quot;")

    # auto refresh page per TTL
    return f"""
    <html>
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1"/>
        <meta http-equiv="refresh" content="{CACHE_TTL_SECONDS}">
        <title>WeatherEdge USA</title>
        <style>
          :root {{
            --bg: #0b1020;
            --text: #eaeefc;
            --muted: rgba(234,238,252,.72);
            --border: rgba(255,255,255,.10);
            --blue: #7aa2ff;
            --shadow: 0 10px 30px rgba(0,0,0,.35);
            --radius: 18px;
          }}
          * {{ box-sizing: border-box; }}
          body {{
            margin: 0;
            background: radial-gradient(1200px 700px at 20% 0%, rgba(122,162,255,.18), transparent 60%),
                        radial-gradient(1000px 600px at 90% 10%, rgba(34,197,94,.14), transparent 55%),
                        var(--bg);
            color: var(--text);
            font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
          }}
          .wrap {{ max-width: 1040px; margin: 0 auto; padding: 22px 16px 42px; }}
          .top {{ display:flex; align-items:flex-end; justify-content:space-between; gap:12px; margin-bottom: 14px; flex-wrap:wrap; }}
          h1 {{ margin:0; font-size: 28px; }}
          .sub {{ color: var(--muted); font-size: 13px; margin-top: 6px; }}
          .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }}
          .pill {{
            display:inline-flex; align-items:center; gap:10px;
            padding: 8px 12px; border-radius: 999px;
            background: rgba(255,255,255,.06);
            border: 1px solid var(--border);
            color: var(--muted);
            font-size: 13px;
          }}
          .badge {{
            padding: 6px 10px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,.14);
            background: rgba(255,255,255,.06);
            font-size: 12px;
            color: var(--text);
          }}
          .grid {{ display:grid; grid-template-columns: repeat(12, 1fr); gap: 14px; margin-top: 14px; }}
          .card {{
            grid-column: span 12;
            background: linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.03));
            border: 1px solid var(--border);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            padding: 14px 14px;
          }}
          .card h2 {{ margin: 0 0 10px; font-size: 16px; color: var(--muted); font-weight: 600; }}
          .big {{ display:flex; align-items:baseline; justify-content:space-between; gap: 12px; flex-wrap: wrap; }}
          .metric {{
            display:flex; flex-direction:column; gap:6px;
            padding: 10px 12px;
            border-radius: 14px;
            background: rgba(0,0,0,.18);
            border: 1px solid rgba(255,255,255,.08);
            min-width: 220px;
          }}
          .metric .label {{ color: var(--muted); font-size: 12px; }}
          .metric .value {{ font-size: 22px; font-weight: 800; }}
          .metric .hint {{ color: var(--muted); font-size: 12px; }}
          table {{ width: 100%; border-collapse: collapse; border-radius: 14px; overflow: hidden; }}
          .th {{ text-align:left; padding: 10px 10px; color: var(--muted); font-size: 12px; border-bottom: 1px solid rgba(255,255,255,.10); }}
          .td {{ padding: 10px 10px; border-bottom: 1px solid rgba(255,255,255,.06); font-size: 13px; }}
          .tag {{
            display:inline-flex; align-items:center; justify-content:center;
            padding: 4px 10px; border-radius: 999px;
            border: 1px solid rgba(255,255,255,.16);
            background: rgba(255,255,255,.04);
            font-size: 12px;
          }}
          .notes {{ margin: 10px 0 0; padding-left: 18px; color: var(--muted); font-size: 13px; }}
          .links a {{ color: var(--blue); text-decoration:none; }}
          .links a:hover {{ text-decoration: underline; }}
          .search {{
            display:flex; gap:10px; flex-wrap:wrap;
            align-items:center;
            padding: 10px; border-radius: 14px;
            border: 1px solid rgba(255,255,255,.10);
            background: rgba(255,255,255,.04);
          }}
          .search input {{
            flex:1; min-width: 220px;
            padding: 10px 12px;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,.14);
            background: rgba(0,0,0,.20);
            color: var(--text);
            outline: none;
          }}
          .search button {{
            padding: 10px 14px;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,.14);
            background: rgba(122,162,255,.16);
            color: var(--text);
            cursor: pointer;
          }}
          .search .hint {{ color: var(--muted); font-size: 12px; }}
          @media (min-width: 860px) {{
            .card.one {{ grid-column: span 7; }}
            .card.two {{ grid-column: span 5; }}
          }}
        </style>
      </head>

      <body>
        <div class="wrap">
          <div class="top">
            <div>
              <h1>WeatherEdge (USA)</h1>
              <div class="sub">
                Location: <strong>{d.location_name}</strong>
                · <span class="mono">{d.lat:.4f},{d.lon:.4f}</span>
                · TZ: <span class="mono">{d.timezone_name}</span>
              </div>
              <div class="sub">
                Tomorrow (local): <strong>{d.local_tomorrow_date}</strong>
                · Cache key: <span class="mono">{key}</span>
                · Cache hit: <strong>{str(hit).lower()}</strong>
              </div>
            </div>
            <div class="pill">
              <span class="badge" style="border-color:{status_color}; color:{status_color};">{status_text}</span>
              <span>Auto refresh: <strong>{REFRESH_MINUTES}m</strong></span>
            </div>
          </div>

          <div class="card">
            <h2>Search anywhere in USA</h2>
            <form class="search" method="get" action="/">
              <input name="q" value="{q_val}" placeholder='City, State (ex: "Los Angeles, CA") or "Honolulu, HI"' />
              <button type="submit">Search</button>
              <div class="hint">Tip: bạn cũng có thể dùng <span class="mono">/?lat=...&lon=...</span></div>
            </form>
          </div>

          <div class="grid">
            <div class="card one">
              <h2>NOW + Last 8 Hours</h2>
              <div class="big">
                <div class="metric">
                  <div class="label">Consensus NOW</div>
                  <div class="value">{fmt_f(d.consensus_now_f)}</div>
                  <div class="hint">Latest datapoint ≤ now</div>
                </div>
                <div class="metric">
                  <div class="label">Consensus 8h MIN / MAX</div>
                  <div class="value">{fmt_f(d.consensus_last8h_min_f)} / {fmt_f(d.consensus_last8h_max_f)}</div>
                  <div class="hint">Window: last 8 hours (UTC)</div>
                </div>
              </div>
              <div style="margin-top:12px;">
                <table>
                  <thead>
                    <tr>
                      <th class="th">Source</th>
                      <th class="th">Status</th>
                      <th class="th">NOW</th>
                      <th class="th">8h MIN</th>
                      <th class="th">8h MAX</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows_now}
                  </tbody>
                </table>
              </div>
            </div>

            <div class="card two">
              <h2>Bet Model</h2>
              <div class="big">
                <div class="metric" style="min-width:100%;">
                  <div class="label">P(TMAX &gt; {d.threshold_f:.0f}°F)</div>
                  <div class="value">{pct(d.p_over) if d.p_over is not None else "N/A"}</div>
                  <div class="hint">
                    Market: {d.market_price*100:.2f}% ({d.market_source})
                    · Edge: {pct(d.edge) if d.edge is not None else "N/A"}
                    · Stake(25% Kelly): {pct(d.stake) if d.stake is not None else "N/A"}
                  </div>
                </div>
              </div>
            </div>

            <div class="card">
              <h2>Tomorrow (Local TZ) MIN / MAX (Sources)</h2>
              <div style="margin-top:6px;">
                <table>
                  <thead>
                    <tr>
                      <th class="th">Source</th>
                      <th class="th">Status</th>
                      <th class="th">Inserted</th>
                      <th class="th">Tomorrow MIN</th>
                      <th class="th">Tomorrow MAX</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows_tmr}
                  </tbody>
                </table>
              </div>

              <div class="sub" style="margin-top:10px;">
                Tomorrow consensus MIN/MAX:
                <strong>{fmt_f(d.consensus_tmr_min_f)} / {fmt_f(d.consensus_tmr_max_f)}</strong>
              </div>

              <div class="sub" style="margin-top:6px;">
                Outliers removed: <span class="mono">{d.removed_outliers or "[]"}</span>
              </div>
              {notes_html}
            </div>

            <div class="card links">
              <h2>API</h2>
              <div class="sub">
                <a href="/api/latest.json">/api/latest.json</a> ·
                <a href="/api/latest.json?q={q_val}">/api/latest.json?q=...</a> ·
                <a href="/api/geocode?q={q_val if q_val else "san%20francisco"}">/api/geocode?q=...</a>
              </div>
              <div class="sub" style="margin-top:8px;">
                Trang auto reload theo TTL. Cache theo location để giảm 429.
              </div>
            </div>
          </div>
        </div>
      </body>
    </html>
    """
