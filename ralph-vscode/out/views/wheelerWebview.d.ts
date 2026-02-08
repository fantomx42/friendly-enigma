import * as vscode from "vscode";
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
export declare class WheelerWebviewProvider implements vscode.WebviewViewProvider {
    private view?;
    private pollInterval?;
    resolveWebviewView(webviewView: vscode.WebviewView): void;
    /** Trigger an immediate content refresh (e.g. when a new memory is stored). */
    refresh(): void;
    private updateContent;
    private readMemory;
    private buildHtml;
    private escapeHtml;
    dispose(): void;
}
