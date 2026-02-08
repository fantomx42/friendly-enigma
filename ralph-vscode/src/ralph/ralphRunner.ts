import { ChildProcess, spawn } from "child_process";
import { EventEmitter } from "events";
import * as path from "path";
import * as vscode from "vscode";
import { RalphState, ParsedEvent } from "../types";
import * as config from "../config";
import { OutputParser } from "./outputParser";

/**
 * Manages the ralph_loop.sh subprocess lifecycle.
 *
 * Emits:
 *   "state"  — RalphState changes
 *   "event"  — ParsedEvent from stdout
 *   "output" — raw stdout text for the output channel
 */
export class RalphRunner extends EventEmitter {
    private proc: ChildProcess | null = null;
    private parser = new OutputParser();
    private _state: RalphState = RalphState.Idle;
    private _objective = "";
    private _iteration = 0;
    private _maxIterations = 0;
    private _model = "";
    private _wheelerCount = 0;

    constructor(private extensionPath: string) {
        super();
        this.parser.on("event", (evt: ParsedEvent) => this.onParsedEvent(evt));
    }

    get state(): RalphState {
        return this._state;
    }
    get objective(): string {
        return this._objective;
    }
    get iteration(): number {
        return this._iteration;
    }
    get maxIterations(): number {
        return this._maxIterations;
    }
    get model(): string {
        return this._model;
    }
    get wheelerCount(): number {
        return this._wheelerCount;
    }

    setObjective(objective: string): void {
        this._objective = objective;
    }

    /** Locate the ai_tech_stack directory relative to the workspace. */
    private findScriptDir(): string {
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

    start(): void {
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

        this.proc = spawn("bash", args, {
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

        this.setState(RalphState.Running);

        this.proc.stdout?.on("data", (data: Buffer) => {
            const text = data.toString();
            this.emit("output", text);
            this.parser.feed(text);
        });

        this.proc.stderr?.on("data", (data: Buffer) => {
            this.emit("output", data.toString());
        });

        this.proc.on("close", (code) => {
            this.parser.flush();
            this.proc = null;
            if (this._state === RalphState.Running) {
                this.setState(
                    code === 0 ? RalphState.Complete : RalphState.Error
                );
            }
        });

        this.proc.on("error", (err) => {
            this.emit("output", `[Extension] Failed to start: ${err.message}\n`);
            this.proc = null;
            this.setState(RalphState.Error);
        });
    }

    stop(): void {
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
        this.setState(RalphState.Idle);
    }

    pauseResume(): void {
        if (!this.proc || !this.proc.pid) {
            return;
        }
        if (this._state === RalphState.Running) {
            this.proc.kill("SIGSTOP");
            this.setState(RalphState.Paused);
        } else if (this._state === RalphState.Paused) {
            this.proc.kill("SIGCONT");
            this.setState(RalphState.Running);
        }
    }

    private setState(state: RalphState): void {
        this._state = state;
        this.emit("state", state);
    }

    private onParsedEvent(evt: ParsedEvent): void {
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
                this.setState(RalphState.Complete);
                vscode.window.showInformationMessage(
                    "Ralph AI: Objective complete!"
                );
                break;
            case "error":
                this.setState(RalphState.Error);
                break;
            case "max_iterations":
                this.setState(RalphState.Error);
                break;
        }
        this.emit("event", evt);
    }

    // TODO(human): Implement error recovery strategy
    handleRecoverableError(_evt: ParsedEvent): void {
        // Decide what to do when Ralph hits a non-fatal error mid-loop.
        // Options: auto-restart, prompt user, log and continue, etc.
    }

    dispose(): void {
        this.stop();
        this.removeAllListeners();
    }
}
