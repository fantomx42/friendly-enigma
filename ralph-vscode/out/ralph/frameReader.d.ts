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
export declare function readNpyFrame(filePath: string): Float32Array | null;
/**
 * Convert a float32 frame to RGBA pixel data for canvas rendering.
 * Maps [-1, 1] to a blue-black-red diverging colormap.
 */
export declare function frameToRGBA(frame: Float32Array, width: number, height: number): Uint8Array;
export interface FrameData {
    id: string;
    pixels: string;
    width: number;
    height: number;
}
/**
 * Load all frames from ~/.wheeler_memory/frames/ and convert to
 * base64 RGBA for webview rendering.
 */
export declare function loadAllFrames(): FrameData[];
