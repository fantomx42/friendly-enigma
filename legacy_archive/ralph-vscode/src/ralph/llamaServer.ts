import { ChildProcess, spawn } from "child_process";
import * as http from "http";
import { EventEmitter } from "events";
import * as vscode from "vscode";
import { LlamaServerState } from "../types";
import * as config from "../config";

/**
 * Manages an optional local llama-server process.
 *
 * Spawns llama-server with settings from VS Code config.
 * All GPU-specific env vars (HSA_OVERRIDE_GFX_VERSION, etc.)
 * are inherited from the user's shell environment.
 *
 * Emits "state" with LlamaServerState on changes.
 */
export class LlamaServer extends EventEmitter {
    private proc: ChildProcess | null = null;
    private _state: LlamaServerState = LlamaServerState.Stopped;
    private healthInterval: ReturnType<typeof setInterval> | null = null;

    get state(): LlamaServerState {
        return this._state;
    }

    /** Check if a server is already running on the configured port. */
    async isExternallyRunning(): Promise<boolean> {
        return this.healthCheck();
    }

    async start(): Promise<boolean> {
        if (this.proc) {
            vscode.window.showWarningMessage("llama-server is already managed.");
            return this._state === LlamaServerState.Ready;
        }

        // Check for externally running server first
        if (await this.isExternallyRunning()) {
            this.setState(LlamaServerState.Ready);
            this.startHealthPolling();
            return true;
        }

        this.setState(LlamaServerState.Starting);

        const modelPath = config.getLlamaServerModelPath();
        const port = config.getLlamaServerPort();
        const ngl = config.getLlamaServerGpuLayers();
        const ctxSize = config.getLlamaServerCtxSize();
        const threads = config.getLlamaServerThreads();

        this.proc = spawn(
            "llama-server",
            [
                "--model",
                modelPath,
                "--mmap",
                "-ngl",
                String(ngl),
                "--ctx-size",
                String(ctxSize),
                "--cache-type-k",
                "q8_0",
                "--cache-type-v",
                "q8_0",
                "--threads",
                String(threads),
                "--port",
                String(port),
            ],
            {
                env: {
                    ...process.env,
                },
            }
        );

        this.proc.on("close", () => {
            this.proc = null;
            this.stopHealthPolling();
            if (this._state !== LlamaServerState.Stopped) {
                this.setState(LlamaServerState.Error);
            }
        });

        this.proc.on("error", (err) => {
            vscode.window.showErrorMessage(
                `llama-server failed to start: ${err.message}`
            );
            this.proc = null;
            this.setState(LlamaServerState.Error);
        });

        // Wait for health endpoint (up to 300s, matching launch.sh)
        const ready = await this.waitForReady(300);
        if (ready) {
            this.setState(LlamaServerState.Ready);
            this.startHealthPolling();
        } else {
            this.setState(LlamaServerState.Error);
            this.stop();
        }
        return ready;
    }

    stop(): void {
        this.stopHealthPolling();
        if (this.proc) {
            this.proc.kill("SIGTERM");
            this.proc = null;
        }
        this.setState(LlamaServerState.Stopped);
    }

    private async waitForReady(timeoutSeconds: number): Promise<boolean> {
        const deadline = Date.now() + timeoutSeconds * 1000;
        while (Date.now() < deadline) {
            if (await this.healthCheck()) {
                return true;
            }
            await this.sleep(2000);
        }
        return false;
    }

    private healthCheck(): Promise<boolean> {
        const port = config.getLlamaServerPort();
        return new Promise((resolve) => {
            const req = http.get(
                `http://127.0.0.1:${port}/health`,
                { timeout: 3000 },
                (res) => {
                    resolve(res.statusCode === 200);
                }
            );
            req.on("error", () => resolve(false));
            req.on("timeout", () => {
                req.destroy();
                resolve(false);
            });
        });
    }

    private startHealthPolling(): void {
        this.stopHealthPolling();
        this.healthInterval = setInterval(async () => {
            const ok = await this.healthCheck();
            if (!ok && this._state === LlamaServerState.Ready) {
                this.setState(LlamaServerState.Error);
            } else if (ok && this._state === LlamaServerState.Error) {
                this.setState(LlamaServerState.Ready);
            }
        }, 30000);
    }

    private stopHealthPolling(): void {
        if (this.healthInterval) {
            clearInterval(this.healthInterval);
            this.healthInterval = null;
        }
    }

    private setState(state: LlamaServerState): void {
        this._state = state;
        this.emit("state", state);
    }

    private sleep(ms: number): Promise<void> {
        return new Promise((resolve) => setTimeout(resolve, ms));
    }

    dispose(): void {
        this.stop();
        this.removeAllListeners();
    }
}
