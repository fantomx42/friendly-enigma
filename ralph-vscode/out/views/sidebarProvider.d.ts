import * as vscode from "vscode";
import { RalphRunner } from "../ralph/ralphRunner";
import { LlamaServer } from "../ralph/llamaServer";
/**
 * TreeDataProvider for the Ralph AI sidebar panel.
 *
 * Displays: objective, run state, iteration progress,
 * model info, Wheeler Memory recall count, llama-server status.
 */
export declare class SidebarProvider implements vscode.TreeDataProvider<SidebarItem> {
    private runner;
    private llamaServer;
    private _onDidChangeTreeData;
    readonly onDidChangeTreeData: vscode.Event<SidebarItem | undefined>;
    constructor(runner: RalphRunner, llamaServer: LlamaServer);
    refresh(): void;
    getTreeItem(element: SidebarItem): vscode.TreeItem;
    getChildren(): SidebarItem[];
    dispose(): void;
}
declare class SidebarItem extends vscode.TreeItem {
    readonly label: string;
    readonly collapsibleState: vscode.TreeItemCollapsibleState;
    constructor(label: string, collapsibleState: vscode.TreeItemCollapsibleState);
}
export {};
