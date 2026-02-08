import { EventEmitter } from "events";
import { RalphState, ParsedEvent } from "../types";
/**
 * Manages the ralph_loop.sh subprocess lifecycle.
 *
 * Emits:
 *   "state"  — RalphState changes
 *   "event"  — ParsedEvent from stdout
 *   "output" — raw stdout text for the output channel
 */
export declare class RalphRunner extends EventEmitter {
    private extensionPath;
    private proc;
    private parser;
    private _state;
    private _objective;
    private _iteration;
    private _maxIterations;
    private _model;
    private _wheelerCount;
    constructor(extensionPath: string);
    get state(): RalphState;
    get objective(): string;
    get iteration(): number;
    get maxIterations(): number;
    get model(): string;
    get wheelerCount(): number;
    setObjective(objective: string): void;
    /** Locate the ai_tech_stack directory relative to the workspace. */
    private findScriptDir;
    start(): void;
    stop(): void;
    pauseResume(): void;
    private setState;
    private onParsedEvent;
    handleRecoverableError(_evt: ParsedEvent): void;
    dispose(): void;
}
