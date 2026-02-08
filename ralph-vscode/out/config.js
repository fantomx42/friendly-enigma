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
exports.getModel = getModel;
exports.getNumCtx = getNumCtx;
exports.getMaxIterations = getMaxIterations;
exports.getOllamaHost = getOllamaHost;
exports.getOutputDir = getOutputDir;
exports.getLlamaServerEnabled = getLlamaServerEnabled;
exports.getLlamaServerModelPath = getLlamaServerModelPath;
exports.getLlamaServerPort = getLlamaServerPort;
exports.getLlamaServerGpuLayers = getLlamaServerGpuLayers;
exports.getLlamaServerCtxSize = getLlamaServerCtxSize;
exports.getLlamaServerThreads = getLlamaServerThreads;
const vscode = __importStar(require("vscode"));
function cfg() {
    return vscode.workspace.getConfiguration("ralph-ai");
}
function getModel() {
    return cfg().get("model", "qwen3:8b");
}
function getNumCtx() {
    return cfg().get("numCtx", 32768);
}
function getMaxIterations() {
    return cfg().get("maxIterations", 50);
}
function getOllamaHost() {
    return cfg().get("ollamaHost", "http://localhost:11434");
}
function getOutputDir() {
    return cfg().get("outputDir", "./ralph_output");
}
function getLlamaServerEnabled() {
    return cfg().get("llamaServer.enabled", false);
}
function getLlamaServerModelPath() {
    const raw = cfg().get("llamaServer.modelPath", "~/VoidAI/models/Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf");
    return raw.replace(/^~/, process.env.HOME || "");
}
function getLlamaServerPort() {
    return cfg().get("llamaServer.port", 8000);
}
function getLlamaServerGpuLayers() {
    return cfg().get("llamaServer.gpuLayers", 18);
}
function getLlamaServerCtxSize() {
    return cfg().get("llamaServer.ctxSize", 8192);
}
function getLlamaServerThreads() {
    return cfg().get("llamaServer.threads", 16);
}
//# sourceMappingURL=config.js.map