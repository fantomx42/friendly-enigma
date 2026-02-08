import { EventEmitter } from "events";
/**
 * State machine that parses ralph_simple.py stdout into typed events.
 *
 * ralph_simple.py emits these patterns on stdout:
 *   >>> Iteration N / MAX
 *   [Wheeler] Recalled N memory(s)
 *   [Wheeler] No memories found
 *   [Wheeler] Stored iteration output
 *   [Ollama:MODEL] Streaming...
 *   <promise>COMPLETE</promise>
 *   [ERROR] ...
 *   Max iterations reached (N)
 *
 * Between "[Ollama:...] Streaming..." and the next "[Wheeler] Storing..."
 * line, all output is raw token data from the model.
 */
export declare class OutputParser extends EventEmitter {
    private streaming;
    private lineBuf;
    parseLine(line: string): void;
    /**
     * Feed raw chunk data (which may contain partial lines).
     * Buffers until newlines arrive, then dispatches complete lines.
     */
    feed(chunk: string): void;
    /** Flush any remaining buffered data. */
    flush(): void;
    reset(): void;
}
