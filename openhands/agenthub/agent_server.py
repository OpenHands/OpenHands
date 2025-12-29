from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()

@app.get("/alive")
def alive():
    return {"status": "ok"}

if __name__ == "__main__":
    import os
    import sys
    port = 3002
    # Allow --port argument to override default
    for i, arg in enumerate(sys.argv):
        if arg == "--port" and i + 1 < len(sys.argv):
            try:
                port = int(sys.argv[i + 1])
            except Exception:
                pass
    port = int(os.environ.get("PORT", port))
    uvicorn.run("openhands.agenthub.agent_server:app", host="0.0.0.0", port=port)
