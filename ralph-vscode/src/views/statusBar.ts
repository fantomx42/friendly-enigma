import * as vscode from "vscode";
import { RalphState } from "../types";
import { RalphRunner } from "../ralph/ralphRunner";

/**
 * Status bar item that shows Ralph's current state.
 * Click opens the output channel.
 */
export class StatusBar {
    private item: vscode.StatusBarItem;

    constructor(private runner: RalphRunner) {
        this.item = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Left,
            100
        );
        this.item.command = "ralph-ai.showOutput";
        this.update(RalphState.Idle);
        this.item.show();

        runner.on("state", (state: RalphState) => this.update(state));
        runner.on("event", () => {
            if (this.runner.state === RalphState.Running) {
                this.item.text = `$(play-circle) Ralph [${this.runner.iteration}/${this.runner.maxIterations}]`;
            }
        });
    }

    private update(state: RalphState): void {
        const display: Record<RalphState, { icon: string; text: string }> = {
            [RalphState.Idle]: { icon: "circle-outline", text: "Ralph: Idle" },
            [RalphState.Running]: {
                icon: "play-circle",
                text: `Ralph [${this.runner.iteration}/${this.runner.maxIterations}]`,
            },
            [RalphState.Paused]: { icon: "debug-pause", text: "Ralph: Paused" },
            [RalphState.Complete]: { icon: "check", text: "Ralph: Complete" },
            [RalphState.Error]: { icon: "error", text: "Ralph: Error" },
        };
        const d = display[state];
        this.item.text = `$(${d.icon}) ${d.text}`;
    }

    dispose(): void {
        this.item.dispose();
    }
}
