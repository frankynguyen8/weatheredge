from fastapi import FastAPI
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse

from app.main import run_once_text

app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/api/run")
def api_run():
    # Luôn trả text; không để provider lỗi làm sập endpoint
    try:
        out = run_once_text()
        return PlainTextResponse(out + "\n")
    except Exception as e:
        # vẫn trả 200 để không bị coi là service down
        return PlainTextResponse(f"Service degraded: {e}\n", status_code=200)

@app.get("/api/run.json")
def api_run_json():
    try:
        out = run_once_text()
        return {"ok": True, "output": out.splitlines()}
    except Exception as e:
        # vẫn trả 200; ok=False để client biết degraded
        return JSONResponse({"ok": False, "error": str(e), "output": []}, status_code=200)

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
      <head><meta name="viewport" content="width=device-width, initial-scale=1"/></head>
      <body style="font-family:-apple-system,Arial;padding:16px;">
        <h2>WeatherEdge</h2>
        <p><a href="/api/run">Run now (text)</a></p>
        <p><a href="/api/run.json">Run now (json)</a></p>
        <p><a href="/health">Health</a></p>
      </body>
    </html>
    """
