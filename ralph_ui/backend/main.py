from fastapi import FastAPI, WebSocket, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import os
import sys
import asyncio
import subprocess
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "../..")
FRONTEND_DIR = os.path.join(BASE_DIR, "../frontend")
LOG_FILE = os.path.join(ROOT_DIR, "ralph.log")
SCRIPT_PATH = os.path.join(ROOT_DIR, "ralph_loop.sh")
METRICS_FILE = os.path.join(ROOT_DIR, "metrics.jsonl")
PLAN_FILE = os.path.join(ROOT_DIR, "RALPH_PLAN.json")

# Ensure ralph_core is importable if needed, but we can read the file directly
sys.path.append(ROOT_DIR)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
async def read_root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/api/metrics")
async def get_metrics():
    """Returns the last 100 metrics."""
    metrics = []
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, "r") as f:
                lines = f.readlines()
                # Get last 100 lines
                for line in lines[-100:]:
                    try:
                        metrics.append(json.loads(line))
                    except:
                        pass
        except Exception as e:
            return {"error": str(e)}
    return metrics

@app.get("/api/plan")
async def get_plan():
    """Returns the current task plan."""
    if os.path.exists(PLAN_FILE):
        try:
            with open(PLAN_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            return {"error": str(e)}
    return {"tasks": []}

async def stream_logs(websocket: WebSocket):
    """
    Continuously reads the log file and sends updates to the client.
    """
    try:
        with open(LOG_FILE, 'r') as f:
            # Move to end of file to start? Or start from beginning?
            # For now, let's start from beginning since we clear it on new command.
            while True:
                line = f.readline()
                if line:
                    await websocket.send_text(line.strip())
                else:
                    await asyncio.sleep(0.1)
    except Exception as e:
        print(f"Log Stream Error: {e}")

async def monitor_process(process, websocket: WebSocket):
    """Waits for the process to finish and signals the client."""
    try:
        # Run the blocking wait() in a separate thread
        await asyncio.to_thread(process.wait)
        await websocket.send_text("SYSTEM::COMPLETE")
    except Exception as e:
        print(f"Monitor Error: {e}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("Ralph Online. Awaiting directive...")
    
    # Ensure log file exists
    if not os.path.exists(LOG_FILE):
        open(LOG_FILE, 'w').close()

    stream_task = None

    try:
        while True:
            # Wait for user input
            data = await websocket.receive_text()
            
            # Heartbeat handling - Do not trigger Ralph
            if data == "PING":
                await websocket.send_text("PONG")
                continue
            
            # 1. Clear previous log
            open(LOG_FILE, 'w').close()
            
            # 2. Kill previous stream if active
            if stream_task:
                stream_task.cancel()
            
            # 3. Start Streamer
            stream_task = asyncio.create_task(stream_logs(websocket))
            
            # 4. Start Ralph
            print(f"Launching Ralph: {data}")
            process = subprocess.Popen(
                ["bash", SCRIPT_PATH, data],
                cwd=ROOT_DIR,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # 5. Monitor for completion
            asyncio.create_task(monitor_process(process, websocket))
            
            await websocket.send_text(f"> Directive Accepted: {data}")
            
    except Exception as e:
        print(f"WebSocket Error: {e}")
    finally:
        if stream_task:
            stream_task.cancel()
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)