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
exports.readNpyFrame = readNpyFrame;
exports.frameToRGBA = frameToRGBA;
exports.loadAllFrames = loadAllFrames;
const fs = __importStar(require("fs"));
const path = __importStar(require("path"));
const FRAMES_DIR = path.join(process.env.HOME || "", ".wheeler_memory", "frames");
/**
 * Parse a NumPy .npy file containing a 128x128 float32 frame.
 *
 * .npy v1.0 layout:
 *   6 bytes  magic  \x93NUMPY
 *   1 byte   major version
 *   1 byte   minor version
 *   2 bytes  header length (little-endian uint16)
 *   N bytes  ASCII header (Python dict literal with shape, dtype, order)
 *   rest     raw data (float32 little-endian for '<f4')
 *
 * Returns flat Float32Array of pixel values in [-1, 1].
 */
function readNpyFrame(filePath) {
    try {
        const buf = fs.readFileSync(filePath);
        // Verify magic
        if (buf[0] !== 0x93 ||
            buf[1] !== 0x4e || // N
            buf[2] !== 0x55 || // U
            buf[3] !== 0x4d || // M
            buf[4] !== 0x50 || // P
            buf[5] !== 0x59 // Y
        ) {
            return null;
        }
        // Header length (little-endian uint16 at offset 8)
        const headerLen = buf[8] | (buf[9] << 8);
        const dataOffset = 10 + headerLen;
        // Read raw float32 data
        const dataBuffer = buf.subarray(dataOffset);
        return new Float32Array(dataBuffer.buffer, dataBuffer.byteOffset, dataBuffer.byteLength / 4);
    }
    catch {
        return null;
    }
}
/**
 * Convert a float32 frame to RGBA pixel data for canvas rendering.
 * Maps [-1, 1] to a blue-black-red diverging colormap.
 */
function frameToRGBA(frame, width, height) {
    const rgba = new Uint8Array(width * height * 4);
    for (let i = 0; i < frame.length; i++) {
        const v = frame[i]; // [-1, 1]
        const offset = i * 4;
        if (v >= 0) {
            // Positive: black → cyan/green
            const t = Math.min(v, 1);
            rgba[offset] = 0; // R
            rgba[offset + 1] = Math.floor(t * 255); // G
            rgba[offset + 2] = Math.floor(t * 180); // B
        }
        else {
            // Negative: black → magenta/red
            const t = Math.min(-v, 1);
            rgba[offset] = Math.floor(t * 255); // R
            rgba[offset + 1] = 0; // G
            rgba[offset + 2] = Math.floor(t * 120); // B
        }
        rgba[offset + 3] = 255; // Alpha
    }
    return rgba;
}
/**
 * Load all frames from ~/.wheeler_memory/frames/ and convert to
 * base64 RGBA for webview rendering.
 */
function loadAllFrames() {
    const results = [];
    if (!fs.existsSync(FRAMES_DIR)) {
        return results;
    }
    const files = fs.readdirSync(FRAMES_DIR).filter((f) => f.endsWith(".npy"));
    for (const file of files) {
        const filePath = path.join(FRAMES_DIR, file);
        const frame = readNpyFrame(filePath);
        if (!frame) {
            continue;
        }
        // Assume 128x128 (standard Wheeler frame size)
        const width = 128;
        const height = 128;
        const rgba = frameToRGBA(frame, width, height);
        results.push({
            id: file.replace(".npy", ""),
            pixels: Buffer.from(rgba).toString("base64"),
            width,
            height,
        });
    }
    return results;
}
//# sourceMappingURL=frameReader.js.map