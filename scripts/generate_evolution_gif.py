#!/usr/bin/env python3
"""Generate a 3-colour CA evolution GIF for Wheeler Memory documentation.

Colour scheme:
  - Blue  (#4A90D9): positive state (value > 0.33)
  - Red   (#D94A4A): negative state (value < -0.33)
  - Dark  (#1A1A2E): neutral  state (otherwise)

Usage:
    python scripts/generate_evolution_gif.py

Output:
    docs/assets/diagrams/evolution.gif
"""

import struct
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# CA dynamics (self-contained so no package install needed)
# ---------------------------------------------------------------------------

def apply_ca_dynamics(frame: np.ndarray) -> np.ndarray:
    """Single CA iteration using 3-state logic (von Neumann neighbourhood)."""
    n_up    = np.roll(frame,  1, axis=0)
    n_down  = np.roll(frame, -1, axis=0)
    n_left  = np.roll(frame,  1, axis=1)
    n_right = np.roll(frame, -1, axis=1)

    is_max = (frame >= n_up) & (frame >= n_down) & (frame >= n_left) & (frame >= n_right)
    is_min = (frame <= n_up) & (frame <= n_down) & (frame <= n_left) & (frame <= n_right)

    neighbors    = np.stack([n_up, n_down, n_left, n_right])
    max_neighbor = np.max(neighbors, axis=0)

    delta = np.zeros_like(frame)
    delta = np.where(is_max,            (1.0 - frame) * 0.35, delta)
    delta = np.where(is_min,           (-1.0 - frame) * 0.35, delta)
    delta = np.where(~is_max & ~is_min, (max_neighbor - frame) * 0.20, delta)

    return np.clip(frame + delta, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Minimal pure-Python GIF encoder
# ---------------------------------------------------------------------------

def _lzw_compress(indices: bytes, min_code_size: int) -> bytes:
    """LZW compress palette indices into GIF image data sub-blocks."""
    clear     = 1 << min_code_size
    eoi       = clear + 1
    table     = {bytes([i]): i for i in range(clear)}
    code_size = min_code_size + 1
    max_code  = 1 << code_size

    bits_out   = 0
    bit_buffer = 0
    raw        = bytearray()

    def emit(code):
        nonlocal bits_out, bit_buffer
        bit_buffer |= code << bits_out
        bits_out   += code_size
        while bits_out >= 8:
            raw.append(bit_buffer & 0xFF)
            bit_buffer >>= 8
            bits_out   -= 8

    emit(clear)
    buf       = b''
    next_code = eoi + 1

    for byte in indices:
        cand = buf + bytes([byte])
        if cand in table:
            buf = cand
        else:
            emit(table[buf])
            if next_code < 4096:
                table[cand] = next_code
                next_code  += 1
                if next_code > max_code and code_size < 12:
                    code_size += 1
                    max_code   = 1 << code_size
            else:
                emit(clear)
                table     = {bytes([i]): i for i in range(clear)}
                next_code = eoi + 1
                code_size = min_code_size + 1
                max_code  = 1 << code_size
            buf = bytes([byte])

    emit(table[buf])
    emit(eoi)
    if bits_out:
        raw.append(bit_buffer & 0xFF)

    # Package into GIF sub-blocks (max 255 bytes each)
    result = bytearray([min_code_size])
    data   = bytes(raw)
    for i in range(0, len(data), 255):
        block = data[i : i + 255]
        result.append(len(block))
        result.extend(block)
    result.append(0)   # block terminator
    return bytes(result)


def _u16le(n: int) -> bytes:
    return struct.pack('<H', n)


def encode_gif(frames: list, palette: list, delay_cs: int = 8) -> bytes:
    """
    Encode a list of 2-D index arrays into an animated GIF89a.

    Parameters
    ----------
    frames   : list of (H, W) uint8 numpy arrays with values in 0..len(palette)-1
    palette  : list of (R, G, B) tuples; length must be a power of 2
    delay_cs : frame delay in centiseconds
    """
    H, W      = frames[0].shape
    n_colors  = len(palette)
    ct_field  = (n_colors.bit_length() - 1) - 1   # log2(n)-1

    gif = bytearray()

    # Header
    gif += b'GIF89a'
    gif += _u16le(W)
    gif += _u16le(H)
    gif += bytes([0x80 | ct_field, 0x00, 0x00])   # global CT present

    # Global Colour Table
    for r, g, b in palette:
        gif += bytes([r, g, b])

    # Netscape 2.0 extension  (infinite loop)
    gif += bytes([
        0x21, 0xFF, 0x0B,
        *b'NETSCAPE2.0',
        0x03, 0x01,
        0x00, 0x00,   # repeat count = 0 → infinite
        0x00,
    ])

    for frame in frames:
        # Graphic Control Extension
        gif += bytes([
            0x21, 0xF9, 0x04,
            0b00000100,          # disposal = restore to background
            *_u16le(delay_cs),
            0x00,                # no transparent index
            0x00,
        ])
        # Image Descriptor
        gif += bytes([0x2C])
        gif += _u16le(0)
        gif += _u16le(0)
        gif += _u16le(W)
        gif += _u16le(H)
        gif += bytes([0x00])    # no local CT, not interlaced

        # Image Data
        min_cs  = max(2, ct_field + 1)
        indices = frame.astype(np.uint8).flatten().tobytes()
        gif    += _lzw_compress(indices, min_cs)

    gif += bytes([0x3B])   # GIF trailer
    return bytes(gif)


# ---------------------------------------------------------------------------
# Colour palette  (4 entries – smallest power-of-2 >= 3)
# ---------------------------------------------------------------------------
# Index 0  neutral  – deep navy (near-black background)
# Index 1  positive – vivid blue
# Index 2  negative – vivid red
# Index 3  padding  (unused)
PALETTE = [
    (26,  26,  46),
    (74, 144, 217),
    (217, 74,  74),
    (0,   0,   0),
]

GRID_SIZE   = 64
PIXEL_SCALE = 5   # each cell → 5×5 px  →  320×320 output
N_TICKS     = 120


def grid_to_frame(grid: np.ndarray) -> np.ndarray:
    """Map continuous [-1,1] values to palette indices, then upscale."""
    idx = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    idx[grid >  0.33] = 1
    idx[grid < -0.33] = 2
    return np.repeat(np.repeat(idx, PIXEL_SCALE, axis=0), PIXEL_SCALE, axis=1)


def main() -> None:
    rng  = np.random.default_rng(42)
    grid = rng.uniform(-1.0, 1.0, (GRID_SIZE, GRID_SIZE))

    history = [grid.copy()]
    for _ in range(N_TICKS):
        grid = apply_ca_dynamics(grid)
        history.append(grid.copy())

    # Include every tick for the first 30, then every 2nd tick
    indices  = list(range(min(30, len(history))))
    indices += list(range(30, len(history), 2))
    frames   = [grid_to_frame(history[i]) for i in indices]

    out = (
        Path(__file__).resolve().parent.parent
        / "docs" / "assets" / "diagrams" / "evolution.gif"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    data = encode_gif(frames, PALETTE, delay_cs=8)
    out.write_bytes(data)

    print(f"Saved  : {out}")
    print(f"Size   : {len(data):,} bytes")
    print(f"Frames : {len(frames)}")
    print(f"Canvas : {GRID_SIZE * PIXEL_SCALE} x {GRID_SIZE * PIXEL_SCALE} px")
    print("Legend : blue = +1 (positive)  |  red = -1 (negative)  |  dark = 0 (neutral)")


if __name__ == "__main__":
    main()
