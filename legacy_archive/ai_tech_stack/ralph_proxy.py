import asyncio
import json
import logging
import os
import subprocess
import sys
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

# Configuration
LLAMA_SERVER_URL = "http://127.0.0.1:8000"
PROXY_PORT = 8001
WHEELER_STORE_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wheeler_store.py")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ralph_proxy")

app = FastAPI()
client = httpx.AsyncClient(base_url=LLAMA_SERVER_URL, timeout=300.0)

async def store_interaction_async(user_prompt: str, assistant_response: str):
    """Call wheeler_store.py in the background."""
    if not user_prompt.strip() or not assistant_response.strip():
        return

    # Create a nice interaction format
    text = "User: " + user_prompt + "\nAssistant: " + assistant_response
    
    cmd = [
        sys.executable,
        WHEELER_STORE_SCRIPT,
        "--text", text,
        "--type", "interaction",
        "--outcome", "success"
    ]
    
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode == 0:
            logger.info(f"Stored interaction: {stdout.decode().strip()}")
        else:
            logger.error(f"Failed to store interaction: {stderr.decode().strip()}")
    except Exception as e:
        logger.error(f"Error calling wheeler_store: {e}")

@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    """Intercept chat completions to log interactions."""
    try:
        body = await request.json()
    except:
        body = {}

    # Extract user prompt (last user message)
    messages = body.get("messages", [])
    user_prompt = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_prompt = m.get("content", "")
            break

    # Forward request to llama-server
    req = client.build_request(
        "POST",
        "/v1/chat/completions",
        json=body,
        timeout=300.0
    )
    r = await client.send(req, stream=True)

    async def stream_wrapper() -> AsyncGenerator[bytes, None]:
        full_response = []
        async for chunk in r.aiter_bytes():
            yield chunk
            # Try to decode chunk to capture response
            try:
                # SSE format: "data: {...}\n\n"
                text_chunk = chunk.decode("utf-8", errors="ignore")
                lines = text_chunk.split("\n")
                for line in lines:
                    if line.startswith("data: ") and line != "data: [DONE]":
                        data_str = line[6:]
                        try:
                            data_json = json.loads(data_str)
                            delta = data_json.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_response.append(content)
                        except json.JSONDecodeError:
                            pass
            except Exception:
                pass
        
        # After stream ends, store memory
        assistant_text = "".join(full_response)
        if user_prompt and assistant_text:
            asyncio.create_task(store_interaction_async(user_prompt, assistant_text))

    return StreamingResponse(
        stream_wrapper(),
        status_code=r.status_code,
        headers=dict(r.headers),
        media_type=r.headers.get("content-type")
    )

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"])
async def proxy_catch_all(request: Request, path: str):
    """Forward all other requests transparently."""
    url = f"/{path}"
    if request.query_params:
        url += f"?{request.query_params}"
    
    content = await request.body()
    
    req = client.build_request(
        request.method,
        url,
        content=content,
        headers=request.headers.raw,
        timeout=300.0
    )
    
    r = await client.send(req, stream=True)
    
    return StreamingResponse(
        r.aiter_bytes(),
        status_code=r.status_code,
        headers=dict(r.headers),
        media_type=r.headers.get("content-type")
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT)