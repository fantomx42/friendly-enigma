import * as vscode from "vscode";
import { RalphState, LlamaServerState } from "../types";
import { RalphRunner } from "../ralph/ralphRunner";
import { LlamaServer } from "../ralph/llamaServer";

/**
 * TreeDataProvider for the Ralph AI sidebar panel.
 *
 * Displays: objective, run state, iteration progress,
 * model info, Wheeler Memory recall count, llama-server status.
 */
export class SidebarProvider implements vscode.TreeDataProvider<SidebarItem> {
    private _onDidChangeTreeData = new vscode.EventEmitter<
        SidebarItem | undefined
    >();
    readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

    constructor(
        private runner: RalphRunner,
        private llamaServer: LlamaServer
    ) {
        runner.on("state", () => this.refresh());
        runner.on("event", () => this.refresh());
        llamaServer.on("state", () => this.refresh());
    }

    refresh(): void {
        this._onDidChangeTreeData.fire(undefined);
    }

    getTreeItem(element: SidebarItem): vscode.TreeItem {
        return element;
    }

    getChildren(): SidebarItem[] {
        const items: SidebarItem[] = [];
        const state = this.runner.state;

        // Objective
        const obj = this.runner.objective || "(none set)";
        const objItem = new SidebarItem(
            `Objective: ${obj}`,
            vscode.TreeItemCollapsibleState.None
        );
        objItem.command = {
            command: "ralph-ai.setObjective",
            title: "Set Objective",
        };
        objItem.iconPath = new vscode.ThemeIcon("target");
        items.push(objItem);

        // State
        const stateIcons: Record<RalphState, string> = {
            [RalphState.Idle]: "circle-outline",
            [RalphState.Running]: "play-circle",
            [RalphState.Paused]: "debug-pause",
            [RalphState.Complete]: "check",
            [RalphState.Error]: "error",
        };
        const stateItem = new SidebarItem(
            `Status: ${state}`,
            vscode.TreeItemCollapsibleState.None
        );
        stateItem.iconPath = new vscode.ThemeIcon(stateIcons[state]);
        items.push(stateItem);

        // Iteration
        if (state !== RalphState.Idle) {
            const iterItem = new SidebarItem(
                `Iteration: ${this.runner.iteration} / ${this.runner.maxIterations}`,
                vscode.TreeItemCollapsibleState.None
            );
            iterItem.iconPath = new vscode.ThemeIcon("sync");
            items.push(iterItem);
        }

        // Model
        if (this.runner.model) {
            const modelItem = new SidebarItem(
                `Model: ${this.runner.model}`,
                vscode.TreeItemCollapsibleState.None
            );
            modelItem.iconPath = new vscode.ThemeIcon("hubot");
            items.push(modelItem);
        }

        // Wheeler Memory
        const wheelerItem = new SidebarItem(
            `Wheeler Memories: ${this.runner.wheelerCount}`,
            vscode.TreeItemCollapsibleState.None
        );
        wheelerItem.iconPath = new vscode.ThemeIcon("database");
        wheelerItem.command = {
            command: "ralph-ai.showWheelerMemory",
            title: "Show Wheeler Memory",
        };
        items.push(wheelerItem);

        // llama-server status
        const llamaIcons: Record<LlamaServerState, string> = {
            [LlamaServerState.Stopped]: "circle-outline",
            [LlamaServerState.Starting]: "loading~spin",
            [LlamaServerState.Ready]: "check",
            [LlamaServerState.Error]: "error",
        };
        const llamaItem = new SidebarItem(
            `llama-server: ${this.llamaServer.state}`,
            vscode.TreeItemCollapsibleState.None
        );
        llamaItem.iconPath = new vscode.ThemeIcon(
            llamaIcons[this.llamaServer.state]
        );
        items.push(llamaItem);

        return items;
    }

    dispose(): void {
        this._onDidChangeTreeData.dispose();
    }
}

class SidebarItem extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState
    ) {
        super(label, collapsibleState);
    }
}
