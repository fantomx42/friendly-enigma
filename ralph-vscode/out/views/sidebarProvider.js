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
exports.SidebarProvider = void 0;
const vscode = __importStar(require("vscode"));
const types_1 = require("../types");
/**
 * TreeDataProvider for the Ralph AI sidebar panel.
 *
 * Displays: objective, run state, iteration progress,
 * model info, Wheeler Memory recall count, llama-server status.
 */
class SidebarProvider {
    runner;
    llamaServer;
    _onDidChangeTreeData = new vscode.EventEmitter();
    onDidChangeTreeData = this._onDidChangeTreeData.event;
    constructor(runner, llamaServer) {
        this.runner = runner;
        this.llamaServer = llamaServer;
        runner.on("state", () => this.refresh());
        runner.on("event", () => this.refresh());
        llamaServer.on("state", () => this.refresh());
    }
    refresh() {
        this._onDidChangeTreeData.fire(undefined);
    }
    getTreeItem(element) {
        return element;
    }
    getChildren() {
        const items = [];
        const state = this.runner.state;
        // Objective
        const obj = this.runner.objective || "(none set)";
        const objItem = new SidebarItem(`Objective: ${obj}`, vscode.TreeItemCollapsibleState.None);
        objItem.command = {
            command: "ralph-ai.setObjective",
            title: "Set Objective",
        };
        objItem.iconPath = new vscode.ThemeIcon("target");
        items.push(objItem);
        // State
        const stateIcons = {
            [types_1.RalphState.Idle]: "circle-outline",
            [types_1.RalphState.Running]: "play-circle",
            [types_1.RalphState.Paused]: "debug-pause",
            [types_1.RalphState.Complete]: "check",
            [types_1.RalphState.Error]: "error",
        };
        const stateItem = new SidebarItem(`Status: ${state}`, vscode.TreeItemCollapsibleState.None);
        stateItem.iconPath = new vscode.ThemeIcon(stateIcons[state]);
        items.push(stateItem);
        // Iteration
        if (state !== types_1.RalphState.Idle) {
            const iterItem = new SidebarItem(`Iteration: ${this.runner.iteration} / ${this.runner.maxIterations}`, vscode.TreeItemCollapsibleState.None);
            iterItem.iconPath = new vscode.ThemeIcon("sync");
            items.push(iterItem);
        }
        // Model
        if (this.runner.model) {
            const modelItem = new SidebarItem(`Model: ${this.runner.model}`, vscode.TreeItemCollapsibleState.None);
            modelItem.iconPath = new vscode.ThemeIcon("hubot");
            items.push(modelItem);
        }
        // Wheeler Memory
        const wheelerItem = new SidebarItem(`Wheeler Memories: ${this.runner.wheelerCount}`, vscode.TreeItemCollapsibleState.None);
        wheelerItem.iconPath = new vscode.ThemeIcon("database");
        wheelerItem.command = {
            command: "ralph-ai.showWheelerMemory",
            title: "Show Wheeler Memory",
        };
        items.push(wheelerItem);
        // llama-server status
        const llamaIcons = {
            [types_1.LlamaServerState.Stopped]: "circle-outline",
            [types_1.LlamaServerState.Starting]: "loading~spin",
            [types_1.LlamaServerState.Ready]: "check",
            [types_1.LlamaServerState.Error]: "error",
        };
        const llamaItem = new SidebarItem(`llama-server: ${this.llamaServer.state}`, vscode.TreeItemCollapsibleState.None);
        llamaItem.iconPath = new vscode.ThemeIcon(llamaIcons[this.llamaServer.state]);
        items.push(llamaItem);
        return items;
    }
    dispose() {
        this._onDidChangeTreeData.dispose();
    }
}
exports.SidebarProvider = SidebarProvider;
class SidebarItem extends vscode.TreeItem {
    label;
    collapsibleState;
    constructor(label, collapsibleState) {
        super(label, collapsibleState);
        this.label = label;
        this.collapsibleState = collapsibleState;
    }
}
//# sourceMappingURL=sidebarProvider.js.map