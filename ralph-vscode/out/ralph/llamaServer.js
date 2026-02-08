"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.LlamaServer = void 0;
const child_process_1 = require("child_process");
const http = __importStar(require("http"));
const events_1 = require("events");
const vscode = __importStar(require("vscode"));
const types_1 = require("../types");
const config = __importStar(require("../config"));
/**
 * Manages an optional local llama-server process.
 *
 * Matches the args from ~/VoidAI/scripts/launch.sh:
 *   --model, --mmap, -ngl 18, --ctx-size 8192,
 *   --cache-type-k q8_0, --cache-type-v q8_0, --threads 16
 *
 * Emits "state" with LlamaServerState on changes.
 */
class LlamaServer extends events_1.EventEmitter {
    proc = null;
    _state = types_1.LlamaServerState.Stopped;
    healthInterval = null;
    get state() {
        return this._state;
    }
    /** Check if a server is already running on the configured port. */
    async isExternallyRunning() {
        return this.healthCheck();
    }
    async start() {
        if (this.proc) {
            vscode.window.showWarningMessage("llama-server is already managed.");
            return this._state === types_1.LlamaServerState.Ready;
        }
        // Check for externally running server first
        if (await this.isExternallyRunning()) {
            this.setState(types_1.LlamaServerState.Ready);
            this.startHealthPolling();
            return true;
        }
        this.setState(types_1.LlamaServerState.Starting);
        const modelPath = config.getLlamaServerModelPath();
        const port = config.getLlamaServerPort();
        const ngl = config.getLlamaServerGpuLayers();
        const ctxSize = config.getLlamaServerCtxSize();
        const threads = config.getLlamaServerThreads();
        this.proc = (0, child_process_1.spawn)("llama-server", [
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
        ], {
            env: {
                ...process.env,
                HSA_OVERRIDE_GFX_VERSION: "12.0.0",
                PYTORCH_ROCM_ARCH: "gfx1200",
            },
        });
        this.proc.on("close", () => {
            this.proc = null;
            this.stopHealthPolling();
            if (this._state !== types_1.LlamaServerState.Stopped) {
                this.setState(types_1.LlamaServerState.Error);
            }
        });
        this.proc.on("error", (err) => {
            vscode.window.showErrorMessage(`llama-server failed to start: ${err.message}`);
            this.proc = null;
            this.setState(types_1.LlamaServerState.Error);
        });
        // Wait for health endpoint (up to 300s, matching launch.sh)
        const ready = await this.waitForReady(300);
        if (ready) {
            this.setState(types_1.LlamaServerState.Ready);
            this.startHealthPolling();
        }
        else {
            this.setState(types_1.LlamaServerState.Error);
            this.stop();
        }
        return ready;
    }
    stop() {
        this.stopHealthPolling();
        if (this.proc) {
            this.proc.kill("SIGTERM");
            this.proc = null;
        }
        this.setState(types_1.LlamaServerState.Stopped);
    }
    async waitForReady(timeoutSeconds) {
        const deadline = Date.now() + timeoutSeconds * 1000;
        while (Date.now() < deadline) {
            if (await this.healthCheck()) {
                return true;
            }
            await this.sleep(2000);
        }
        return false;
    }
    healthCheck() {
        const port = config.getLlamaServerPort();
        return new Promise((resolve) => {
            const req = http.get(`http://127.0.0.1:${port}/health`, { timeout: 3000 }, (res) => {
                resolve(res.statusCode === 200);
            });
            req.on("error", () => resolve(false));
            req.on("timeout", () => {
                req.destroy();
                resolve(false);
            });
        });
    }
    startHealthPolling() {
        this.stopHealthPolling();
        this.healthInterval = setInterval(async () => {
            const ok = await this.healthCheck();
            if (!ok && this._state === types_1.LlamaServerState.Ready) {
                this.setState(types_1.LlamaServerState.Error);
            }
            else if (ok && this._state === types_1.LlamaServerState.Error) {
                this.setState(types_1.LlamaServerState.Ready);
            }
        }, 30000);
    }
    stopHealthPolling() {
        if (this.healthInterval) {
            clearInterval(this.healthInterval);
            this.healthInterval = null;
        }
    }
    setState(state) {
        this._state = state;
        this.emit("state", state);
    }
    sleep(ms) {
        return new Promise((resolve) => setTimeout(resolve, ms));
    }
    dispose() {
        this.stop();
        this.removeAllListeners();
    }
}
exports.LlamaServer = LlamaServer;
//# sourceMappingURL=llamaServer.js.map