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
exports.RalphRunner = void 0;
const child_process_1 = require("child_process");
const events_1 = require("events");
const path = __importStar(require("path"));
const vscode = __importStar(require("vscode"));
const types_1 = require("../types");
const config = __importStar(require("../config"));
const outputParser_1 = require("./outputParser");
/**
 * Manages the ralph_loop.sh subprocess lifecycle.
 *
 * Emits:
 *   "state"  — RalphState changes
 *   "event"  — ParsedEvent from stdout
 *   "output" — raw stdout text for the output channel
 */
class RalphRunner extends events_1.EventEmitter {
    extensionPath;
    proc = null;
    parser = new outputParser_1.OutputParser();
    _state = types_1.RalphState.Idle;
    _objective = "";
    _iteration = 0;
    _maxIterations = 0;
    _model = "";
    _wheelerCount = 0;
    constructor(extensionPath) {
        super();
        this.extensionPath = extensionPath;
        this.parser.on("event", (evt) => this.onParsedEvent(evt));
    }
    get state() {
        return this._state;
    }
    get objective() {
        return this._objective;
    }
    get iteration() {
        return this._iteration;
    }
    get maxIterations() {
        return this._maxIterations;
    }
    get model() {
        return this._model;
    }
    get wheelerCount() {
        return this._wheelerCount;
    }
    setObjective(objective) {
        this._objective = objective;
    }
    /** Locate the ai_tech_stack directory relative to the workspace. */
    findScriptDir() {
        const workspaceFolders = vscode.workspace.workspaceFolders;
        if (workspaceFolders) {
            // Check if we're in the ralph repo
            for (const folder of workspaceFolders) {
                const candidate = path.join(folder.uri.fsPath, "ai_tech_stack");
                return candidate;
            }
        }
        // Fallback: assume extensionPath is inside ralph/ralph-vscode
        return path.join(this.extensionPath, "..", "ai_tech_stack");
    }
    start() {
        if (this.proc) {
            vscode.window.showWarningMessage("Ralph is already running.");
            return;
        }
        if (!this._objective) {
            vscode.window.showWarningMessage("Set an objective first.");
            return;
        }
        const scriptDir = this.findScriptDir();
        const scriptPath = path.join(scriptDir, "ralph_loop.sh");
        const args = [
            scriptPath,
            this._objective,
            "--model",
            config.getModel(),
            "--num-ctx",
            String(config.getNumCtx()),
            "--max-iterations",
            String(config.getMaxIterations()),
            "--output-dir",
            config.getOutputDir(),
        ];
        const ollamaHost = config.getOllamaHost();
        this.parser.reset();
        this._iteration = 0;
        this._maxIterations = config.getMaxIterations();
        this._wheelerCount = 0;
        this.proc = (0, child_process_1.spawn)("bash", args, {
            cwd: scriptDir,
            env: {
                ...process.env,
                RALPH_MODEL: config.getModel(),
                RALPH_NUM_CTX: String(config.getNumCtx()),
                OLLAMA_HOST: ollamaHost,
                HSA_OVERRIDE_GFX_VERSION: "12.0.0",
                PYTHONUNBUFFERED: "1",
            },
        });
        this.setState(types_1.RalphState.Running);
        this.proc.stdout?.on("data", (data) => {
            const text = data.toString();
            this.emit("output", text);
            this.parser.feed(text);
        });
        this.proc.stderr?.on("data", (data) => {
            this.emit("output", data.toString());
        });
        this.proc.on("close", (code) => {
            this.parser.flush();
            this.proc = null;
            if (this._state === types_1.RalphState.Running) {
                this.setState(code === 0 ? types_1.RalphState.Complete : types_1.RalphState.Error);
            }
        });
        this.proc.on("error", (err) => {
            this.emit("output", `[Extension] Failed to start: ${err.message}\n`);
            this.proc = null;
            this.setState(types_1.RalphState.Error);
        });
    }
    stop() {
        if (!this.proc) {
            return;
        }
        this.proc.kill("SIGTERM");
        // Force kill after 5 seconds if still alive
        const pid = this.proc.pid;
        setTimeout(() => {
            if (this.proc && this.proc.pid === pid) {
                this.proc.kill("SIGKILL");
            }
        }, 5000);
        this.setState(types_1.RalphState.Idle);
    }
    pauseResume() {
        if (!this.proc || !this.proc.pid) {
            return;
        }
        if (this._state === types_1.RalphState.Running) {
            this.proc.kill("SIGSTOP");
            this.setState(types_1.RalphState.Paused);
        }
        else if (this._state === types_1.RalphState.Paused) {
            this.proc.kill("SIGCONT");
            this.setState(types_1.RalphState.Running);
        }
    }
    setState(state) {
        this._state = state;
        this.emit("state", state);
    }
    onParsedEvent(evt) {
        switch (evt.type) {
            case "iteration_start":
                this._iteration = evt.iteration || 0;
                this._maxIterations = evt.maxIterations || this._maxIterations;
                break;
            case "wheeler_recall":
                this._wheelerCount = evt.memoryCount || 0;
                break;
            case "wheeler_no_memories":
                this._wheelerCount = 0;
                break;
            case "streaming_start":
                this._model = evt.model || this._model;
                break;
            case "completion":
                this.setState(types_1.RalphState.Complete);
                vscode.window.showInformationMessage("Ralph AI: Objective complete!");
                break;
            case "error":
                this.setState(types_1.RalphState.Error);
                break;
            case "max_iterations":
                this.setState(types_1.RalphState.Error);
                break;
        }
        this.emit("event", evt);
    }
    // TODO(human): Implement error recovery strategy
    handleRecoverableError(_evt) {
        // Decide what to do when Ralph hits a non-fatal error mid-loop.
        // Options: auto-restart, prompt user, log and continue, etc.
    }
    dispose() {
        this.stop();
        this.removeAllListeners();
    }
}
exports.RalphRunner = RalphRunner;
//# sourceMappingURL=ralphRunner.js.map