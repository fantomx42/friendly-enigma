"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.LlamaServerState = exports.RalphState = void 0;
var RalphState;
(function (RalphState) {
    RalphState["Idle"] = "idle";
    RalphState["Running"] = "running";
    RalphState["Paused"] = "paused";
    RalphState["Complete"] = "complete";
    RalphState["Error"] = "error";
})(RalphState || (exports.RalphState = RalphState = {}));
var LlamaServerState;
(function (LlamaServerState) {
    LlamaServerState["Stopped"] = "stopped";
    LlamaServerState["Starting"] = "starting";
    LlamaServerState["Ready"] = "ready";
    LlamaServerState["Error"] = "error";
})(LlamaServerState || (exports.LlamaServerState = LlamaServerState = {}));
//# sourceMappingURL=types.js.map