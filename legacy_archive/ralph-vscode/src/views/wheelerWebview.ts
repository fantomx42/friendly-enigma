import * as vscode from "vscode";
import * as fs from "fs";
import * as path from "path";
import { WheelerMemoryData, WheelerMemoryEntry } from "../types";
import { loadAllFrames, FrameData } from "../ralph/frameReader";

const MEMORY_PATH = path.join(
    process.env.HOME || "",
    ".wheeler_memory",
    "memory.json"
);

/**
 * WebviewViewProvider that shows Wheeler Memory frames as visual heatmaps.
 *
 * Reads ~/.wheeler_memory/memory.json for metadata and
 * ~/.wheeler_memory/frames/*.npy for the actual 128x128 spatial
 * encodings. Renders each frame on an HTML5 Canvas with a
 * diverging colormap (negative=magenta, zero=black, positive=cyan).
 *
 * Polls every 10s while visible.
 */
export class WheelerWebviewProvider implements vscode.WebviewViewProvider {
    private view?: vscode.WebviewView;
    private pollInterval?: ReturnType<typeof setInterval>;

    resolveWebviewView(webviewView: vscode.WebviewView): void {
        this.view = webviewView;

        webviewView.webview.options = { enableScripts: true };
        this.updateContent();

        this.pollInterval = setInterval(() => {
            if (this.view?.visible) {
                this.updateContent();
            }
        }, 10000);

        webviewView.onDidDispose(() => {
            if (this.pollInterval) {
                clearInterval(this.pollInterval);
            }
        });
    }

    /** Trigger an immediate content refresh (e.g. when a new memory is stored). */
    refresh(): void {
        this.updateContent();
    }

    private updateContent(): void {
        if (!this.view) {
            return;
        }

        const data = this.readMemory();
        const memories = data?.memories || [];
        const recent = memories
            .sort((a, b) => b.timestamp - a.timestamp)
            .slice(0, 10);

        const frames = loadAllFrames();

        this.view.webview.html = this.buildHtml(recent, memories.length, frames);
    }

    private readMemory(): WheelerMemoryData | null {
        try {
            const raw = fs.readFileSync(MEMORY_PATH, "utf-8");
            return JSON.parse(raw) as WheelerMemoryData;
        } catch {
            return null;
        }
    }

    private buildHtml(
        entries: WheelerMemoryEntry[],
        totalCount: number,
        frames: FrameData[]
    ): string {
        // Build memory cards with matched frames
        const cards = entries
            .map((e) => {
                const date = new Date(e.timestamp * 1000).toLocaleString();
                const preview = this.escapeHtml(e.text);
                const stability = (e.stability * 100).toFixed(0);
                const frame = frames.find((f) => f.id === e.id);
                const canvasId = `frame-${e.id}`;

                const frameHtml = frame
                    ? `<canvas id="${canvasId}" width="${frame.width}" height="${frame.height}" class="frame-canvas"></canvas>`
                    : `<div class="no-frame">No frame</div>`;

                return `<div class="card">
                    <div class="frame-container">
                        ${frameHtml}
                    </div>
                    <div class="card-body">
                        <div class="meta">
                            <span class="stability">${stability}% stable</span>
                            <span class="hits">${e.hits} hits</span>
                            <span class="type">${this.escapeHtml(e.type)}</span>
                        </div>
                        <div class="preview">${preview}</div>
                        <div class="date">${date}</div>
                    </div>
                </div>`;
            })
            .join("\n");

        // Unmatched frames (frames without memory.json entries)
        const matchedIds = new Set(entries.map((e) => e.id));
        const orphanFrames = frames.filter((f) => !matchedIds.has(f.id));
        const orphanCards = orphanFrames
            .map(
                (f) => `<div class="card orphan">
                <div class="frame-container">
                    <canvas id="frame-${f.id}" width="${f.width}" height="${f.height}" class="frame-canvas"></canvas>
                </div>
                <div class="card-body">
                    <div class="meta"><span class="type">orphan frame</span></div>
                    <div class="preview">${f.id}</div>
                </div>
            </div>`
            )
            .join("\n");

        // Serialize frame data for the script
        const frameDataJson = JSON.stringify(
            frames.map((f) => ({ id: f.id, pixels: f.pixels, w: f.width, h: f.height }))
        );

        return `<!DOCTYPE html>
<html>
<head>
<style>
    * { box-sizing: border-box; }
    body {
        font-family: var(--vscode-font-family);
        font-size: var(--vscode-font-size);
        color: var(--vscode-foreground);
        background: var(--vscode-sideBar-background);
        padding: 8px;
        margin: 0;
    }
    .header {
        font-weight: bold;
        margin-bottom: 10px;
        padding-bottom: 6px;
        border-bottom: 1px solid var(--vscode-panel-border);
        color: var(--vscode-foreground);
    }
    .header .count {
        color: var(--vscode-charts-green);
    }
    .card {
        border: 1px solid var(--vscode-panel-border);
        border-radius: 6px;
        margin-bottom: 10px;
        overflow: hidden;
        background: var(--vscode-editor-background);
    }
    .frame-container {
        background: #000;
        display: flex;
        justify-content: center;
        padding: 4px;
    }
    .frame-canvas {
        width: 100%;
        max-width: 256px;
        height: auto;
        image-rendering: pixelated;
        border: 1px solid #333;
    }
    .no-frame {
        width: 100%;
        height: 64px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #666;
        font-style: italic;
    }
    .card-body {
        padding: 8px;
    }
    .meta {
        display: flex;
        gap: 8px;
        font-size: 0.82em;
        margin-bottom: 4px;
    }
    .stability { color: var(--vscode-charts-green); font-weight: bold; }
    .hits { color: var(--vscode-charts-blue); }
    .type {
        color: var(--vscode-charts-yellow);
        background: rgba(255,255,0,0.08);
        padding: 0 4px;
        border-radius: 3px;
    }
    .preview {
        font-size: 0.9em;
        white-space: pre-wrap;
        word-break: break-word;
        color: var(--vscode-foreground);
        opacity: 0.85;
        margin: 4px 0;
        max-height: 180px;
        overflow-y: auto;
        padding: 6px;
        background: rgba(0, 0, 0, 0.15);
        border-radius: 4px;
    }
    .date {
        font-size: 0.75em;
        color: var(--vscode-descriptionForeground);
    }
    .orphan { opacity: 0.6; }
    .empty {
        color: var(--vscode-descriptionForeground);
        font-style: italic;
        text-align: center;
        padding: 20px;
    }
    .section-label {
        font-size: 0.8em;
        color: var(--vscode-descriptionForeground);
        margin: 12px 0 6px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
</head>
<body>
    <div class="header">
        Wheeler Memory &mdash; <span class="count">${totalCount}</span> memories, <span class="count">${frames.length}</span> frames
    </div>

    ${cards || '<div class="empty">No Wheeler memories found.<br>Run Ralph to generate spatial encodings.</div>'}

    ${orphanCards ? `<div class="section-label">Orphan Frames</div>${orphanCards}` : ""}

    <script>
        // Render frame pixel data onto canvases
        const frames = ${frameDataJson};
        for (const f of frames) {
            const canvas = document.getElementById('frame-' + f.id);
            if (!canvas) continue;
            const ctx = canvas.getContext('2d');
            if (!ctx) continue;

            // Decode base64 RGBA
            const binary = atob(f.pixels);
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) {
                bytes[i] = binary.charCodeAt(i);
            }

            const imageData = ctx.createImageData(f.w, f.h);
            imageData.data.set(bytes);
            ctx.putImageData(imageData, 0, 0);
        }
    </script>
</body>
</html>`;
    }

    private escapeHtml(s: string): string {
        return s
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;");
    }

    dispose(): void {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
        }
    }
}
