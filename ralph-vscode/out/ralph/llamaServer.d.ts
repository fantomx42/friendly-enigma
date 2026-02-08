import { EventEmitter } from "events";
import { LlamaServerState } from "../types";
/**
 * Manages an optional local llama-server process.
 *
 * Matches the args from ~/VoidAI/scripts/launch.sh:
 *   --model, --mmap, -ngl 18, --ctx-size 8192,
 *   --cache-type-k q8_0, --cache-type-v q8_0, --threads 16
 *
 * Emits "state" with LlamaServerState on changes.
 */
export declare class LlamaServer extends EventEmitter {
    private proc;
    private _state;
    private healthInterval;
    get state(): LlamaServerState;
    /** Check if a server is already running on the configured port. */
    isExternallyRunning(): Promise<boolean>;
    start(): Promise<boolean>;
    stop(): void;
    private waitForReady;
    private healthCheck;
    private startHealthPolling;
    private stopHealthPolling;
    private setState;
    private sleep;
    dispose(): void;
}
