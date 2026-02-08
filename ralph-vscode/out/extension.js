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
exports.activate = activate;
exports.deactivate = deactivate;
const vscode = __importStar(require("vscode"));
const path = __importStar(require("path"));
const ralphRunner_1 = require("./ralph/ralphRunner");
const llamaServer_1 = require("./ralph/llamaServer");
const sidebarProvider_1 = require("./views/sidebarProvider");
const statusBar_1 = require("./views/statusBar");
const wheelerWebview_1 = require("./views/wheelerWebview");
const config = __importStar(require("./config"));
let runner;
let llamaServer;
let statusBar;
let outputChannel;
function activate(context) {
    outputChannel = vscode.window.createOutputChannel("Ralph AI");
    runner = new ralphRunner_1.RalphRunner(context.extensionPath);
    llamaServer = new llamaServer_1.LlamaServer();
    // Pipe raw stdout to the output channel
    runner.on("output", (text) => {
        outputChannel.append(text);
    });
    // Sidebar tree view
    const sidebarProvider = new sidebarProvider_1.SidebarProvider(runner, llamaServer);
    vscode.window.registerTreeDataProvider("ralph-ai.sidebar", sidebarProvider);
    // Wheeler Memory webview
    const wheelerProvider = new wheelerWebview_1.WheelerWebviewProvider();
    context.subscriptions.push(vscode.window.registerWebviewViewProvider("ralph-ai.wheeler", wheelerProvider));
    // Refresh Wheeler webview immediately when a new memory is stored
    runner.on("event", (evt) => {
        if (evt.type === "wheeler_stored") {
            wheelerProvider.refresh();
        }
    });
    // Status bar
    statusBar = new statusBar_1.StatusBar(runner);
    // Commands
    context.subscriptions.push(vscode.commands.registerCommand("ralph-ai.setObjective", async () => {
        const objective = await vscode.window.showInputBox({
            prompt: "Enter the objective for Ralph AI",
            placeHolder: "Describe what Ralph should accomplish...",
            value: runner.objective,
        });
        if (objective !== undefined) {
            runner.setObjective(objective);
            sidebarProvider.refresh();
        }
    }), vscode.commands.registerCommand("ralph-ai.start", async () => {
        // Auto-start llama-server if configured
        if (config.getLlamaServerEnabled()) {
            const ready = await llamaServer.start();
            if (!ready) {
                vscode.window.showErrorMessage("llama-server failed to start. Ralph not started.");
                return;
            }
        }
        runner.start();
        outputChannel.show(true);
    }), vscode.commands.registerCommand("ralph-ai.stop", () => {
        runner.stop();
    }), vscode.commands.registerCommand("ralph-ai.pauseResume", () => {
        runner.pauseResume();
    }), vscode.commands.registerCommand("ralph-ai.showOutput", () => {
        outputChannel.show(true);
    }), vscode.commands.registerCommand("ralph-ai.openIterationOutput", async () => {
        const outputDir = config.getOutputDir();
        const workspaceFolders = vscode.workspace.workspaceFolders;
        let base = outputDir;
        if (workspaceFolders &&
            !path.isAbsolute(outputDir)) {
            base = path.join(workspaceFolders[0].uri.fsPath, "ai_tech_stack", outputDir);
        }
        const iteration = runner.iteration || 1;
        const filePath = path.join(base, `iteration_${iteration}_output.txt`);
        try {
            const doc = await vscode.workspace.openTextDocument(vscode.Uri.file(filePath));
            await vscode.window.showTextDocument(doc);
        }
        catch {
            vscode.window.showWarningMessage(`Output file not found: ${filePath}`);
        }
    }), vscode.commands.registerCommand("ralph-ai.startLlamaServer", () => {
        llamaServer.start();
    }), vscode.commands.registerCommand("ralph-ai.stopLlamaServer", () => {
        llamaServer.stop();
    }), vscode.commands.registerCommand("ralph-ai.showWheelerMemory", () => {
        // Focus the Wheeler webview panel in the sidebar
        vscode.commands.executeCommand("ralph-ai.wheeler.focus");
    }));
    // Disposables
    context.subscriptions.push({
        dispose() {
            runner.dispose();
            llamaServer.dispose();
            statusBar.dispose();
            sidebarProvider.dispose();
            wheelerProvider.dispose();
            outputChannel.dispose();
        },
    });
}
function deactivate() {
    runner?.dispose();
    llamaServer?.dispose();
}
//# sourceMappingURL=extension.js.map