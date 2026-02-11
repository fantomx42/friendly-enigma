import { EventEmitter } from "events";
import { ParsedEvent } from "../types";

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
export class OutputParser extends EventEmitter {
    private streaming = false;
    private lineBuf = "";

    parseLine(line: string): void {
        // Iteration header
        const iterMatch = line.match(/^>>> Iteration (\d+) \/ (\d+)/);
        if (iterMatch) {
            this.streaming = false;
            this.emit("event", {
                type: "iteration_start",
                iteration: parseInt(iterMatch[1], 10),
                maxIterations: parseInt(iterMatch[2], 10),
            } as ParsedEvent);
            return;
        }

        // Wheeler recall
        const recallMatch = line.match(
            /^\[Wheeler\] Recalled (\d+) memory\(s\)/
        );
        if (recallMatch) {
            this.emit("event", {
                type: "wheeler_recall",
                memoryCount: parseInt(recallMatch[1], 10),
            } as ParsedEvent);
            return;
        }

        // Wheeler no memories
        if (line.includes("[Wheeler] No memories found")) {
            this.emit("event", {
                type: "wheeler_no_memories",
                memoryCount: 0,
            } as ParsedEvent);
            return;
        }

        // Wheeler stored
        if (line.includes("[Wheeler] Stored iteration output")) {
            this.streaming = false;
            this.emit("event", { type: "wheeler_stored" } as ParsedEvent);
            return;
        }

        // Streaming start
        const streamMatch = line.match(/^\[Ollama:(.+?)\] Streaming/);
        if (streamMatch) {
            this.streaming = true;
            this.emit("event", {
                type: "streaming_start",
                model: streamMatch[1],
            } as ParsedEvent);
            return;
        }

        // Completion promise
        if (line.includes("<promise>COMPLETE</promise>")) {
            this.emit("event", { type: "completion" } as ParsedEvent);
            // Don't return — the line may also contain token data
        }

        // Error
        const errMatch = line.match(/^\[ERROR\]\s*(.*)/);
        if (errMatch) {
            this.streaming = false;
            this.emit("event", {
                type: "error",
                message: errMatch[1],
            } as ParsedEvent);
            return;
        }

        // Max iterations
        const maxMatch = line.match(/^Max iterations reached \((\d+)\)/);
        if (maxMatch) {
            this.streaming = false;
            this.emit("event", {
                type: "max_iterations",
                iteration: parseInt(maxMatch[1], 10),
            } as ParsedEvent);
            return;
        }

        // Raw token data while streaming
        if (this.streaming) {
            this.emit("event", {
                type: "token",
                token: line,
            } as ParsedEvent);
        }
    }

    /**
     * Feed raw chunk data (which may contain partial lines).
     * Buffers until newlines arrive, then dispatches complete lines.
     */
    feed(chunk: string): void {
        this.lineBuf += chunk;
        const lines = this.lineBuf.split("\n");
        // Keep the last (possibly incomplete) segment in the buffer
        this.lineBuf = lines.pop() || "";
        for (const line of lines) {
            if (line.length > 0) {
                this.parseLine(line);
            }
        }
    }

    /** Flush any remaining buffered data. */
    flush(): void {
        if (this.lineBuf.length > 0) {
            this.parseLine(this.lineBuf);
            this.lineBuf = "";
        }
    }

    reset(): void {
        this.streaming = false;
        this.lineBuf = "";
    }
}
