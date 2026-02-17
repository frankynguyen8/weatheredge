from fastapi import FastAPI
from app.main import run_once_text
from fastapi.responses import HTMLResponse
import subprocess, os

app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/api/run")
def run_once():

from app.main import run_once_text

@app.get("/api/run")
def api_run():
    try:
        out = run_once_text()
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
      <head><meta name="viewport" content="width=device-width, initial-scale=1"/></head>
      <body style="font-family: -apple-system, Arial; padding:16px;">
        <h2>WeatherEdge</h2>
        <p><a href="/api/run">Run now</a> (chạy 1 lần và xem kết quả)</p>
        <p><a href="/health">Health</a></p>
      </body>
    </html>
    """
