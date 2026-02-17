from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from app.main import run_once_text   # IMPORT ĐÚNG PATH

app = FastAPI()


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/api/run")
def api_run():
    try:
        output = run_once_text()
        return {
            "ok": True,
            "output": output
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e)
        }


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <meta name="viewport" content="width=device-width, initial-scale=1" />
        </head>
        <body style="font-family: Arial; padding:20px;">
            <h2>WeatherEdge</h2>
            <p><a href="/api/run">Run now</a></p>
            <p><a href="/health">Health</a></p>
        </body>
    </html>
    """
