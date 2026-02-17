import asyncio
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse

from app.main import run_once_struct, run_once_text

app = FastAPI()

# cache in memory
CACHE = {
    "updated_at_utc": None,
    "data": None,
    "error": None,
}

REFRESH_SECONDS = 30 * 60  # 30 minutes


def pct(x):
    return None if x is None else round(x * 100, 2)


def fmt_f(x):
    return "N/A" if x is None else f"{x:.1f}°F"


async def refresh_loop():
    while True:
        try:
            data = run_once_struct()
            CACHE["data"] = data
            CACHE["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            CACHE["error"] = None
        except Exception as e:
            CACHE["error"] = str(e)
        await asyncio.sleep(REFRESH_SECONDS)


@app.on_event("startup")
async def on_startup():
    # run first refresh immediately + then loop
    try:
        data = run_once_struct()
        CACHE["data"] = data
        CACHE["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        CACHE["error"] = None
    except Exception as e:
        CACHE["error"] = str(e)

    asyncio.create_task(refresh_loop())


@app.get("/health")
def health():
    return {"ok": True, "cache_updated_at_utc": CACHE["updated_at_utc"], "error": CACHE["error"]}


@app.get("/api/run", response_class=PlainTextResponse)
def api_run():
    # on-demand text (still safe)
    try:
        return run_once_text() + "\n"
    except Exception as e:
        return PlainTextResponse(f"Service degraded: {e}\n", status_code=200)


@app.get("/api/latest.json")
def latest_json():
    # cached structured result
    if not CACHE["data"]:
        return JSONResponse({"ok": False, "error": CACHE["error"], "data": None}, status_code=200)

    d = CACHE["data"]
    return {
        "ok": True,
        "cache_updated_at_utc": CACHE["updated_at_utc"],
        "generated_at_utc": d.generated_at_utc,
        "lat": d.lat,
        "lon": d.lon,
        "ny_date_tomorrow": d.ny_date_tomorrow,
        "consensus_min_f": d.consensus_min,
        "consensus_max_f": d.consensus_max,
        "threshold_f": d.threshold_f,
        "p_over": d.p_over,
        "market_price": d.market_price,
        "edge": d.edge,
        "stake": d.stake,
        "removed_outliers": d.removed_outliers,
        "notes": d.notes,
        "sources": [
            {
                "src": s.src,
                "rows_inserted": s.rows_inserted,
                "tomorrow_min_f": s.tmr_min,
                "tomorrow_max_f": s.tmr_max,
                "next_hours": s.next_hours,
            } for s in d.sources
        ],
    }


@app.get("/", response_class=HTMLResponse)
def home():
    d = CACHE["data"]
    updated = CACHE["updated_at_utc"]
    err = CACHE["error"]

    if not d:
        return f"""
        <html><head><meta name="viewport" content="width=device-width, initial-scale=1"/></head>
        <body style="font-family:ui-sans-serif,system-ui;padding:20px;background:#0b1020;color:#eaeefc;">
          <h2 style="margin:0 0 8px;">WeatherEdge</h2>
          <p style="opacity:.8;margin:0 0 16px;">Cache chưa có dữ liệu.</p>
          <pre style="background:#121a33;padding:12px;border-radius:12px;overflow:auto;">{err or "No error info"}</pre>
          <p><a style="color:#7aa2ff" href="/api/latest.json">/api/latest.json</a></p>
          <p><a style="color:#7aa2ff" href="/health">/health</a></p>
        </body></html>
        """

    # badges
    p_over = d.p_over
    market = d.market_price
    edge = d.edge
    stake = d.stake

    # choose simple status color
    status_color = "#22c55e" if (edge is not None and edge > 0) else "#f97316"
    status_text = "EDGE+" if (edge is not None and edge > 0) else "NO EDGE"

    # build sources table
    rows = ""
    for s in d.sources:
        rows += f"""
        <tr>
          <td class="td mono">{s.src}</td>
          <td class="td mono">{s.rows_inserted}</td>
          <td class="td">{fmt_f(s.tmr_min)}</td>
          <td class="td">{fmt_f(s.tmr_max)}</td>
        </tr>
        """

    # notes
    notes_html = ""
    if d.notes:
        notes_html = "<ul class='notes'>" + "".join([f"<li>{n}</li>" for n in d.notes]) + "</ul>"

    # auto refresh page every 30 minutes
    return f"""
    <html>
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1"/>
        <meta http-equiv="refresh" content="{REFRESH_SECONDS}">
        <title>WeatherEdge</title>
        <style>
          :root {{
            --bg: #0b1020;
            --panel: #0f1730;
            --card: #121a33;
            --text: #eaeefc;
            --muted: rgba(234,238,252,.72);
            --border: rgba(255,255,255,.10);
            --blue: #7aa2ff;
            --green: #22c55e;
            --orange: #f97316;
            --red: #ef4444;
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
          .wrap {{ max-width: 980px; margin: 0 auto; padding: 22px 16px 42px; }}
          .top {{
            display:flex; align-items:flex-end; justify-content:space-between; gap:12px;
            margin-bottom: 14px;
          }}
          h1 {{ margin:0; font-size: 28px; letter-spacing: .2px; }}
          .sub {{ color: var(--muted); font-size: 13px; margin-top: 6px; }}
          .pill {{
            display:inline-flex; align-items:center; gap:8px;
            padding: 8px 12px; border-radius: 999px;
            background: rgba(255,255,255,.06);
            border: 1px solid var(--border);
            color: var(--muted);
            font-size: 13px;
          }}
          .pill strong {{ color: var(--text); }}
          .grid {{
            display:grid;
            grid-template-columns: repeat(12, 1fr);
            gap: 14px;
            margin-top: 14px;
          }}
          .card {{
            grid-column: span 12;
            background: linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.03));
            border: 1px solid var(--border);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            padding: 14px 14px;
          }}
          .card h2 {{ margin: 0 0 10px; font-size: 16px; color: var(--muted); font-weight: 600; }}
          .big {{
            display:flex; align-items:baseline; justify-content:space-between; gap: 12px;
            flex-wrap: wrap;
          }}
          .metric {{
            display:flex; flex-direction:column; gap:6px;
            padding: 10px 12px;
            border-radius: 14px;
            background: rgba(0,0,0,.18);
            border: 1px solid rgba(255,255,255,.08);
            min-width: 220px;
          }}
          .metric .label {{ color: var(--muted); font-size: 12px; }}
          .metric .value {{ font-size: 22px; font-weight: 800; letter-spacing: .2px; }}
          .metric .hint {{ color: var(--muted); font-size: 12px; }}
          .badge {{
            padding: 6px 10px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,.14);
            background: rgba(255,255,255,.06);
            font-size: 12px;
            color: var(--text);
          }}
          .badge.status {{
            border-color: rgba(255,255,255,.14);
            background: rgba(255,255,255,.06);
          }}
          table {{
            width: 100%;
            border-collapse: collapse;
            overflow: hidden;
            border-radius: 14px;
          }}
          .th {{
            text-align:left;
            padding: 10px 10px;
            color: var(--muted);
            font-size: 12px;
            border-bottom: 1px solid rgba(255,255,255,.10);
          }}
          .td {{
            padding: 10px 10px;
            border-bottom: 1px solid rgba(255,255,255,.06);
            font-size: 13px;
          }}
          .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }}
          .notes {{ margin: 10px 0 0; padding-left: 18px; color: var(--muted); font-size: 13px; }}
          .links a {{ color: var(--blue); text-decoration:none; }}
          .links a:hover {{ text-decoration: underline; }}
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
              <h1>WeatherEdge</h1>
              <div class="sub">
                Tomorrow (NY): <strong>{d.ny_date_tomorrow}</strong> ·
                Location: <span class="mono">{d.lat:.5f},{d.lon:.5f}</span>
              </div>
              <div class="sub">
                Cache updated (UTC): <span class="mono">{updated}</span>
                · Generated (UTC): <span class="mono">{d.generated_at_utc}</span>
              </div>
            </div>
            <div class="pill">
              <span class="badge status" style="border-color:{status_color}; color:{status_color};">{status_text}</span>
              <span>Auto refresh: <strong>30m</strong></span>
            </div>
          </div>

          <div class="grid">
            <div class="card one">
              <h2>Tomorrow (NY) MIN / MAX (Consensus)</h2>
              <div class="big">
                <div class="metric">
                  <div class="label">Consensus MIN</div>
                  <div class="value">{fmt_f(d.consensus_min)}</div>
                  <div class="hint">Avg after outlier filter</div>
                </div>
                <div class="metric">
                  <div class="label">Consensus MAX</div>
                  <div class="value">{fmt_f(d.consensus_max)}</div>
                  <div class="hint">Avg after outlier filter</div>
                </div>
              </div>
            </div>

            <div class="card two">
              <h2>Bet Model</h2>
              <div class="big">
                <div class="metric" style="min-width:100%;">
                  <div class="label">P(TMAX &gt; {d.threshold_f:.0f}°F)</div>
                  <div class="value">{("N/A" if d.p_over is None else f"{pct(d.p_over):.2f}%")}</div>
                  <div class="hint">
                    Market: {("N/A" if market is None else f"{pct(market):.2f}%")}
                    · Edge: {("N/A" if edge is None else f"{pct(edge):.2f}%")}
                    · Stake(25% Kelly): {("N/A" if stake is None else f"{pct(stake):.2f}%")}
                  </div>
                </div>
              </div>
            </div>

            <div class="card">
              <h2>Sources (Tomorrow NY MIN/MAX)</h2>
              <table>
                <thead>
                  <tr>
                    <th class="th">Source</th>
                    <th class="th">Inserted</th>
                    <th class="th">Tomorrow MIN</th>
                    <th class="th">Tomorrow MAX</th>
                  </tr>
                </thead>
                <tbody>
                  {rows}
                </tbody>
              </table>
              <div class="sub" style="margin-top:10px;">
                Outliers removed: <span class="mono">{d.removed_outliers or "[]"}</span>
              </div>
              {notes_html}
            </div>

            <div class="card links">
              <h2>API</h2>
              <div class="sub">
                <a href="/api/latest.json">/api/latest.json</a> ·
                <a href="/api/run">/api/run</a> ·
                <a href="/health">/health</a>
              </div>
              <div class="sub" style="margin-top:8px;">
                Trang tự reload mỗi 30 phút. Background cũng refresh cache mỗi 30 phút.
              </div>
            </div>
          </div>
        </div>
      </body>
    </html>
    """
