import * as vscode from "vscode";
import * as path from "path";
import { spawn } from "child_process";

export class ChatProvider implements vscode.WebviewViewProvider {
    private view?: vscode.WebviewView;
    private history: { role: string; content: string }[] = [];

    constructor(private extensionPath: string) {}

    resolveWebviewView(webviewView: vscode.WebviewView): void {
        this.view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [vscode.Uri.file(path.join(this.extensionPath, "media"))],
        };

        webviewView.webview.html = this.getHtmlContent();

        webviewView.webview.onDidReceiveMessage(async (data) => {
            switch (data.type) {
                case "sendMessage":
                    await this.handleSendMessage(data.text);
                    break;
                case "clearHistory":
                    this.history = [];
                    break;
            }
        });
    }

    private async handleSendMessage(text: string) {
        if (!this.view) return;

        // Add user message to UI immediately
        this.view.webview.postMessage({ type: "addMessage", role: "user", content: text });

        try {
            const response = await this.queryRalph(text);
            
            // Add assistant response to UI
            this.view.webview.postMessage({ 
                type: "addMessage", 
                role: "assistant", 
                content: response.response,
                wheeler: response.wheeler_context 
            });

            // Update local history
            this.history.push({ role: "user", content: text });
            this.history.push({ role: "assistant", content: response.response });

        } catch (err: any) {
            this.view.webview.postMessage({ type: "addMessage", role: "error", content: err.message });
        }
    }

    private async queryRalph(input: string): Promise<any> {
        const workspaceFolders = vscode.workspace.workspaceFolders;
        let scriptPath = "";
        if (workspaceFolders) {
            scriptPath = path.join(workspaceFolders[0].uri.fsPath, "ai_tech_stack", "ralph_chat_api.py");
        }

        const payload = JSON.stringify({
            input: input,
            history: this.history,
            model: vscode.workspace.getConfiguration("ralph-ai").get("model"),
        });

        return new Promise((resolve, reject) => {
            const proc = spawn("python3", [scriptPath, "--json", payload]);
            let output = "";
            let error = "";

            proc.stdout.on("data", (data) => (output += data.toString()));
            proc.stderr.on("data", (data) => (error += data.toString()));

            proc.on("close", (code) => {
                if (code === 0) {
                    try {
                        resolve(JSON.parse(output));
                    } catch (e) {
                        reject(new Error("Failed to parse Ralph output: " + output));
                    }
                } else {
                    reject(new Error(error || `Ralph exited with code ${code}`));
                }
            });
        });
    }

    private getHtmlContent(): string {
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <style>
        body { font-family: var(--vscode-font-family); padding: 10px; display: flex; flex-direction: column; height: 100vh; margin: 0; box-sizing: border-box; }
        #chat-container { flex: 1; overflow-y: auto; margin-bottom: 10px; display: flex; flex-direction: column; gap: 10px; }
        .message { padding: 8px; border-radius: 6px; max-width: 90%; word-wrap: break-word; }
        .user { align-self: flex-end; background: var(--vscode-button-background); color: var(--vscode-button-foreground); }
        .assistant { align-self: flex-start; background: var(--vscode-editor-inactiveSelectionBackground); border: 1px solid var(--vscode-panel-border); }
        .error { align-self: center; color: var(--vscode-errorForeground); font-size: 0.9em; }
        .wheeler-info { font-size: 0.75em; color: var(--vscode-charts-green); margin-top: 4px; border-top: 1px solid rgba(0,0,0,0.1); padding-top: 2px; }
        #input-container { display: flex; gap: 5px; padding-bottom: 10px; }
        input { flex: 1; background: var(--vscode-input-background); color: var(--vscode-input-foreground); border: 1px solid var(--vscode-input-border); padding: 5px; border-radius: 2px; }
        button { background: var(--vscode-button-background); color: var(--vscode-button-foreground); border: none; padding: 5px 10px; cursor: pointer; border-radius: 2px; }
        button:hover { background: var(--vscode-button-hoverBackground); }
    </style>
</head>
<body>
    <div id="chat-container"></div>
    <div id="input-container">
        <input type="text" id="user-input" placeholder="Type a message..." />
        <button id="send-btn">Send</button>
    </div>

    <script>
        const vscode = acquireVsCodeApi();
        const chatContainer = document.getElementById('chat-container');
        const userInput = document.getElementById('user-input');
        const sendBtn = document.getElementById('send-btn');

        function addMessage(role, content, wheeler) {
            const div = document.createElement('div');
            div.className = 'message ' + role;
            
            const textSpan = document.createElement('span');
            textSpan.textContent = content;
            div.appendChild(textSpan);

            if (wheeler) {
                const wheelerDiv = document.createElement('div');
                wheelerDiv.className = 'wheeler-info';
                wheelerDiv.textContent = 'Wheeler recalled: ' + wheeler.substring(0, 50) + '...';
                div.appendChild(wheelerDiv);
            }

            chatContainer.appendChild(div);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        sendBtn.addEventListener('click', () => {
            const text = userInput.value.trim();
            if (text) {
                vscode.postMessage({ type: 'sendMessage', text: text });
                userInput.value = '';
            }
        });

        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendBtn.click();
            }
        });

        window.addEventListener('message', event => {
            const message = event.data;
            if (message.type === 'addMessage') {
                addMessage(message.role, message.content, message.wheeler);
            }
        });
    </script>
</body>
</html>`;
    }
}
