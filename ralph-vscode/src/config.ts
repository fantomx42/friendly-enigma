import * as vscode from "vscode";

function cfg(): vscode.WorkspaceConfiguration {
    return vscode.workspace.getConfiguration("ralph-ai");
}

export function getModel(): string {
    return cfg().get<string>("model", "qwen3:8b");
}

export function getNumCtx(): number {
    return cfg().get<number>("numCtx", 32768);
}

export function getMaxIterations(): number {
    return cfg().get<number>("maxIterations", 50);
}

export function getOllamaHost(): string {
    return cfg().get<string>("ollamaHost", "http://localhost:11434");
}

export function getOutputDir(): string {
    return cfg().get<string>("outputDir", "./ralph_output");
}

export function getLlamaServerEnabled(): boolean {
    return cfg().get<boolean>("llamaServer.enabled", false);
}

export function getLlamaServerModelPath(): string {
    const raw = cfg().get<string>(
        "llamaServer.modelPath",
        "~/VoidAI/models/Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf"
    );
    return raw.replace(/^~/, process.env.HOME || "");
}

export function getLlamaServerPort(): number {
    return cfg().get<number>("llamaServer.port", 8000);
}

export function getLlamaServerGpuLayers(): number {
    return cfg().get<number>("llamaServer.gpuLayers", 18);
}

export function getLlamaServerCtxSize(): number {
    return cfg().get<number>("llamaServer.ctxSize", 8192);
}

export function getLlamaServerThreads(): number {
    return cfg().get<number>("llamaServer.threads", 16);
}
