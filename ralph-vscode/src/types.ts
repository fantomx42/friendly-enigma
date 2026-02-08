export enum RalphState {
    Idle = "idle",
    Running = "running",
    Paused = "paused",
    Complete = "complete",
    Error = "error",
}

export enum LlamaServerState {
    Stopped = "stopped",
    Starting = "starting",
    Ready = "ready",
    Error = "error",
}

export type ParsedEventType =
    | "iteration_start"
    | "wheeler_recall"
    | "wheeler_no_memories"
    | "streaming_start"
    | "token"
    | "completion"
    | "error"
    | "max_iterations"
    | "wheeler_stored";

export interface ParsedEvent {
    type: ParsedEventType;
    iteration?: number;
    maxIterations?: number;
    memoryCount?: number;
    model?: string;
    message?: string;
    token?: string;
}

export interface WheelerMemoryEntry {
    id: string;
    text: string;
    type: string;
    outcome: string;
    summary: string | null;
    errors: string[];
    timestamp: number;
    hits: number;
    stability: number;
}

export interface WheelerMemoryData {
    memories: WheelerMemoryEntry[];
    last_checkpoint: number;
}
