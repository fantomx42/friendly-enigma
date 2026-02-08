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
exports.StatusBar = void 0;
const vscode = __importStar(require("vscode"));
const types_1 = require("../types");
/**
 * Status bar item that shows Ralph's current state.
 * Click opens the output channel.
 */
class StatusBar {
    runner;
    item;
    constructor(runner) {
        this.runner = runner;
        this.item = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
        this.item.command = "ralph-ai.showOutput";
        this.update(types_1.RalphState.Idle);
        this.item.show();
        runner.on("state", (state) => this.update(state));
        runner.on("event", () => {
            if (this.runner.state === types_1.RalphState.Running) {
                this.item.text = `$(play-circle) Ralph [${this.runner.iteration}/${this.runner.maxIterations}]`;
            }
        });
    }
    update(state) {
        const display = {
            [types_1.RalphState.Idle]: { icon: "circle-outline", text: "Ralph: Idle" },
            [types_1.RalphState.Running]: {
                icon: "play-circle",
                text: `Ralph [${this.runner.iteration}/${this.runner.maxIterations}]`,
            },
            [types_1.RalphState.Paused]: { icon: "debug-pause", text: "Ralph: Paused" },
            [types_1.RalphState.Complete]: { icon: "check", text: "Ralph: Complete" },
            [types_1.RalphState.Error]: { icon: "error", text: "Ralph: Error" },
        };
        const d = display[state];
        this.item.text = `$(${d.icon}) ${d.text}`;
    }
    dispose() {
        this.item.dispose();
    }
}
exports.StatusBar = StatusBar;
//# sourceMappingURL=statusBar.js.map