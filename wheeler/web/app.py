import os
import io
import asyncio
from flask import Flask, send_file, render_template_string
from wheeler.core.memory import WheelerMemory
from wheeler.core.viz import render_frame
import matplotlib.pyplot as plt

app = Flask(__name__)

# Global memory instance
MEMORY_DIR = os.environ.get("WHEELER_STORAGE", "./.wheeler")
wm = WheelerMemory(MEMORY_DIR)

# HTML Template
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Wheeler Memory Dashboard</title>
    <style>
        body { font-family: sans-serif; margin: 2rem; background: #1a1a1a; color: #e0e0e0; }
        h1 { color: #bb86fc; }
        .memory-card { 
            background: #2d2d2d; 
            padding: 1rem; 
            margin: 1rem 0; 
            border-radius: 8px; 
            display: flex;
            align-items: center;
        }
        .memory-info { flex: 1; }
        .memory-img { width: 128px; height: 128px; margin-left: 1rem; border: 1px solid #444; }
        a { color: #03dac6; text-decoration: none; }
    </style>
</head>
<body>
    <h1>Wheeler Memory Dashboard</h1>
    <p>Storage: {{ storage }}</p>
    
    <h2>Stored Memories</h2>
    <div id="memories">
        {% for mem in memories %}
        <div class="memory-card">
            <div class="memory-info">
                <h3>{{ mem.key }}</h3>
                <p>UUID: {{ mem.uuid }}</p>
                <p>Stability: {{ "%.4f"|format(mem.stability) }} | Hits: {{ mem.hit_count }}</p>
                <p>Confidence: {{ "%.2f"|format(mem.confidence) }}</p>
            </div>
            <img class="memory-img" src="/image/{{ mem.uuid }}" loading="lazy">
        </div>
        {% endfor %}
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    # Sync wrapper for async call
    memories = asyncio.run(get_memories())[:50]
    return render_template_string(HTML, memories=memories, storage=MEMORY_DIR)

@app.route('/image/<uuid>')
def image(uuid):
    frame = asyncio.run(get_frame(uuid))
    if frame is None:
        return "Not found", 404
        
    # Render to buffer
    buf = io.BytesIO()
    # We use our own simple render logic here to output to buf
    # render_frame expects a path, so we'll adapt it or just plot directly
    plt.figure(figsize=(2, 2))
    plt.imshow(frame, cmap='magma')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    return send_file(buf, mimetype='image/png')

async def get_memories():
    if not os.path.exists(MEMORY_DIR):
        return []
    await wm.initialize()
    return await wm.storage.metadata.list_memories()

async def get_frame(uuid):
    if not os.path.exists(MEMORY_DIR):
        return None
    await wm.initialize()
    mem = await wm.load_by_uuid(uuid)
    if mem:
        return mem['frame']
    return None

def start_server():
    print(f"Starting dashboard on http://127.0.0.1:5000")
    print(f"Reading from: {MEMORY_DIR}")
    app.run(host='127.0.0.1', port=5000)

if __name__ == '__main__':
    start_server()
