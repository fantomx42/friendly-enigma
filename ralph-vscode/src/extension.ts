import * as vscode from "vscode";
import * as path from "path";
import { RalphRunner } from "./ralph/ralphRunner";
import { LlamaServer } from "./ralph/llamaServer";
import { SidebarProvider } from "./views/sidebarProvider";
import { StatusBar } from "./views/statusBar";
import { WheelerWebviewProvider } from "./views/wheelerWebview";
import { ParsedEvent } from "./types";
import * as config from "./config";

let runner: RalphRunner;
let llamaServer: LlamaServer;
let statusBar: StatusBar;
let outputChannel: vscode.OutputChannel;

export function activate(context: vscode.ExtensionContext) {
    outputChannel = vscode.window.createOutputChannel("Ralph AI");

    runner = new RalphRunner(context.extensionPath);
    llamaServer = new LlamaServer();

    // Pipe raw stdout to the output channel
    runner.on("output", (text: string) => {
        outputChannel.append(text);
    });

    // Sidebar tree view
    const sidebarProvider = new SidebarProvider(runner, llamaServer);
    vscode.window.registerTreeDataProvider("ralph-ai.sidebar", sidebarProvider);

    // Wheeler Memory webview
    const wheelerProvider = new WheelerWebviewProvider();
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(
            "ralph-ai.wheeler",
            wheelerProvider
        )
    );

    // Refresh Wheeler webview immediately when a new memory is stored
    runner.on("event", (evt: ParsedEvent) => {
        if (evt.type === "wheeler_stored") {
            wheelerProvider.refresh();
        }
    });

    // Status bar
    statusBar = new StatusBar(runner);

    // Commands
    context.subscriptions.push(
        vscode.commands.registerCommand("ralph-ai.setObjective", async () => {
            const objective = await vscode.window.showInputBox({
                prompt: "Enter the objective for Ralph AI",
                placeHolder: "Describe what Ralph should accomplish...",
                value: runner.objective,
            });
            if (objective !== undefined) {
                runner.setObjective(objective);
                sidebarProvider.refresh();
            }
        }),

        vscode.commands.registerCommand("ralph-ai.start", async () => {
            // Auto-start llama-server if configured
            if (config.getLlamaServerEnabled()) {
                const ready = await llamaServer.start();
                if (!ready) {
                    vscode.window.showErrorMessage(
                        "llama-server failed to start. Ralph not started."
                    );
                    return;
                }
            }
            runner.start();
            outputChannel.show(true);
        }),

        vscode.commands.registerCommand("ralph-ai.stop", () => {
            runner.stop();
        }),

        vscode.commands.registerCommand("ralph-ai.pauseResume", () => {
            runner.pauseResume();
        }),

        vscode.commands.registerCommand("ralph-ai.showOutput", () => {
            outputChannel.show(true);
        }),

        vscode.commands.registerCommand(
            "ralph-ai.openIterationOutput",
            async () => {
                const outputDir = config.getOutputDir();
                const workspaceFolders = vscode.workspace.workspaceFolders;
                let base = outputDir;
                if (
                    workspaceFolders &&
                    !path.isAbsolute(outputDir)
                ) {
                    base = path.join(
                        workspaceFolders[0].uri.fsPath,
                        "ai_tech_stack",
                        outputDir
                    );
                }

                const iteration = runner.iteration || 1;
                const filePath = path.join(
                    base,
                    `iteration_${iteration}_output.txt`
                );
                try {
                    const doc = await vscode.workspace.openTextDocument(
                        vscode.Uri.file(filePath)
                    );
                    await vscode.window.showTextDocument(doc);
                } catch {
                    vscode.window.showWarningMessage(
                        `Output file not found: ${filePath}`
                    );
                }
            }
        ),

        vscode.commands.registerCommand("ralph-ai.startLlamaServer", () => {
            llamaServer.start();
        }),

        vscode.commands.registerCommand("ralph-ai.stopLlamaServer", () => {
            llamaServer.stop();
        }),

        vscode.commands.registerCommand("ralph-ai.showWheelerMemory", () => {
            // Focus the Wheeler webview panel in the sidebar
            vscode.commands.executeCommand("ralph-ai.wheeler.focus");
        })
    );

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

export function deactivate() {
    runner?.dispose();
    llamaServer?.dispose();
}
