# Specification: Visualization & Diagnostics

## Objective
To provide visual insight into the Wheeler Memory system. If the attractors are the "thoughts" of the system, we need a way to see them. This track implements heatmap generation, dynamics animation, and pattern diagnostics.

## Core Components

### 1. Frame Visualizer (`wheeler.core.viz`)
*   **Heatmap Generation:** Convert 128x128 tensors into high-quality heatmaps (PNG/SVG).
*   **Attractor Comparison:** Side-by-side visualization of two attractors with their correlation highlighted.
*   **Evolution Video:** Export the N-tick dynamics process as a GIF or MP4 to see how the system settles.

### 2. Diagnostic Tools
*   **Entropy Mapping:** Visualize the "energy" or "tension" across the grid.
*   **Zone Analysis:** Map which parts of the grid are being used by which characters (TextCodec debug).

## Technical Requirements
*   **Matplotlib/Seaborn:** For static heatmap generation.
*   **Pillow/ImageIO:** For animation/GIF support.
*   **CLI Integration:** `wheeler viz <uuid>` and `wheeler viz-evolution "text"`.

## Success Criteria
*   [ ] Can generate a PNG heatmap of any stored memory by UUID.
*   [ ] Can generate a GIF showing the "settling" of a new input into an attractor.
*   [ ] Diagnostic tool identifies "overcrowded" zones in the grid.
