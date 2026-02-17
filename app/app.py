import asyncio
import os
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse

from app.main import run_once_struct, run_once_text

app = FastAPI()

CACHE = {"updated_at_utc": None, "data": None, "error": None}

REFRESH_MINUTES = int(os.getenv("REFRESH_MINUTES", "30"))
REFRESH_SECONDS = max(60, REFRESH_MINUTES * 60)  # minimum 60s


def pct(x):
    return None if x is None else round(x * 100, 2)


def fmt_f(x):
    return "N/A" if x is None else f"{x:.1f}°F"


def badge_for_status(status: str):
    if status == "OK":
        return ("OK", "#22c55e")
    if status == "NO_KEY":
        return ("NO_KEY", "#f97316")
    return ("EMPTY", "#94a3b8")


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
    try:
        return run_once_text() + "\n"
    except Exception as e:
        return PlainTextResponse(f"Service degraded: {e}\n", status_code=200)


@app.get("/api/latest.json")
def latest_json():
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

        "consensus_now_f": d.consensus_now_f,
        "consensus_last8h_min_f": d.consensus_last8h_min_f,
        "consensus_last8h_max_f": d.consensus_last8h_max_f,

        "consensus_tmr_min_f": d.consensus_tmr_min_f,
        "consensus_tmr_max_f": d.consensus_tmr_max_f,

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

    # status badge based on edge
    edge = d.edge
    status_color = "#22c55e" if (edge is not None and edge > 0) else "#f97316"
    status_text = "EDGE+" if (edge is not None and edge > 0) else "NO EDGE"

    # Build NOW + last8h table
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

    # Build tomorrow table
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
          .wrap {{ max-width: 1020px; margin: 0 auto; padding: 22px 16px 42px; }}
          .top {{
            display:flex; align-items:flex-end; justify-content:space-between; gap:12px;
            margin-bottom: 14px;
          }}
          h1 {{ margin:0; font-size: 28px; }}
          .sub {{ color: var(--muted); font-size: 13px; margin-top: 6px; }}
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
          .metric .value {{ font-size: 22px; font-weight: 800; }}
          .metric .hint {{ color: var(--muted); font-size: 12px; }}
          table {{ width: 100%; border-collapse: collapse; border-radius: 14px; overflow: hidden; }}
          .th {{
            text-align:left; padding: 10px 10px;
            color: var(--muted); font-size: 12px;
            border-bottom: 1px solid rgba(255,255,255,.10);
          }}
          .td {{
            padding: 10px 10px;
            border-bottom: 1px solid rgba(255,255,255,.06);
            font-size: 13px;
          }}
          .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }}
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
              <span class="badge" style="border-color:{status_color}; color:{status_color};">{status_text}</span>
              <span>Auto refresh: <strong>{REFRESH_MINUTES}m</strong></span>
            </div>
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
                  <div class="value">{("N/A" if d.p_over is None else f"{pct(d.p_over):.2f}%")}</div>
                  <div class="hint">
                    Market: {("N/A" if d.market_price is None else f"{pct(d.market_price):.2f}%")}
                    · Edge: {("N/A" if d.edge is None else f"{pct(d.edge):.2f}%")}
                    · Stake(25% Kelly): {("N/A" if d.stake is None else f"{pct(d.stake):.2f}%")}
                  </div>
                </div>
              </div>
            </div>

            <div class="card">
              <h2>Tomorrow (NY) MIN / MAX (Sources)</h2>
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
                <a href="/api/run">/api/run</a> ·
                <a href="/health">/health</a>
              </div>
              <div class="sub" style="margin-top:8px;">
                Trang tự reload theo REFRESH_MINUTES. Background cũng refresh cache theo REFRESH_MINUTES.
              </div>
            </div>
          </div>
        </div>
      </body>
    </html>
    """
