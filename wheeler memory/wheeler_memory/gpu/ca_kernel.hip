/*
 * Wheeler Memory — HIP GPU Kernel for Cellular Automata Evolution
 *
 * Batch-parallel 3-state CA dynamics on AMD GPUs via ROCm/HIP.
 * Each thread handles one cell in one frame. A batch of B frames
 * with 64×64 = 4096 cells uses B × 4096 threads per kernel launch.
 *
 * Build:  hipcc --offload-arch=gfx1201 -shared -fPIC -O3 -o libwheeler_ca.so ca_kernel.hip
 * Fallback: hipcc --offload-arch=gfx1100 ...  (RDNA 3 compat mode)
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstring>

#define GRID_W 64
#define GRID_H 64
#define GRID_SIZE (GRID_W * GRID_H)  /* 4096 */
#define STABILITY_THRESHOLD 1e-4f

/* ─── CA step kernel ──────────────────────────────────────────────────
 * Layout: frames[batch_idx * GRID_SIZE + row * GRID_W + col]
 * One thread per cell per frame.  Wrapping (toroidal) boundaries.
 */
__global__ void ca_step_kernel(const float* __restrict__ frames_in,
                               float* __restrict__ frames_out,
                               float* __restrict__ deltas,
                               int batch_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cells = batch_size * GRID_SIZE;
    if (tid >= total_cells) return;

    int b   = tid / GRID_SIZE;
    int idx = tid % GRID_SIZE;
    int row = idx / GRID_W;
    int col = idx % GRID_W;

    int base = b * GRID_SIZE;

    /* Von Neumann neighbors with toroidal wrap */
    int up_row    = (row == 0)          ? GRID_H - 1 : row - 1;
    int down_row  = (row == GRID_H - 1) ? 0          : row + 1;
    int left_col  = (col == 0)          ? GRID_W - 1 : col - 1;
    int right_col = (col == GRID_W - 1) ? 0          : col + 1;

    float cell  = frames_in[base + row * GRID_W + col];
    float n_up    = frames_in[base + up_row   * GRID_W + col];
    float n_down  = frames_in[base + down_row * GRID_W + col];
    float n_left  = frames_in[base + row      * GRID_W + left_col];
    float n_right = frames_in[base + row      * GRID_W + right_col];

    /* Classify cell role */
    bool is_max = (cell >= n_up) && (cell >= n_down) && (cell >= n_left) && (cell >= n_right);
    bool is_min = (cell <= n_up) && (cell <= n_down) && (cell <= n_left) && (cell <= n_right);

    float delta;
    if (is_max) {
        delta = (1.0f - cell) * 0.35f;
    } else if (is_min) {
        delta = (-1.0f - cell) * 0.35f;
    } else {
        /* Slope: flow toward max neighbor */
        float max_n = fmaxf(fmaxf(n_up, n_down), fmaxf(n_left, n_right));
        delta = (max_n - cell) * 0.20f;
    }

    float new_val = fminf(fmaxf(cell + delta, -1.0f), 1.0f);  /* clip [-1, +1] */
    frames_out[base + row * GRID_W + col] = new_val;

    /* Per-cell absolute delta (for convergence check) */
    deltas[tid] = fabsf(new_val - cell);
}


/* ─── Reduction kernel ────────────────────────────────────────────────
 * Computes mean absolute delta per frame (for convergence detection).
 * One block per frame, shared-memory reduction.
 */
__global__ void reduce_delta_kernel(const float* __restrict__ deltas,
                                    float* __restrict__ mean_deltas,
                                    int batch_size)
{
    int b = blockIdx.x;
    if (b >= batch_size) return;

    __shared__ float sdata[256];

    int tid_local = threadIdx.x;
    int base = b * GRID_SIZE;

    /* Each thread sums multiple elements */
    float sum = 0.0f;
    for (int i = tid_local; i < GRID_SIZE; i += blockDim.x) {
        sum += deltas[base + i];
    }
    sdata[tid_local] = sum;
    __syncthreads();

    /* Tree reduction */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid_local < s) {
            sdata[tid_local] += sdata[tid_local + s];
        }
        __syncthreads();
    }

    if (tid_local == 0) {
        mean_deltas[b] = sdata[0] / (float)GRID_SIZE;
    }
}


/* ─── Host API (C linkage for ctypes) ─────────────────────────────── */

extern "C" {

/**
 * Evolve a batch of frames on GPU.
 *
 * @param frames_in   Input: B × 4096 float32 array (row-major)
 * @param frames_out  Output: B × 4096 float32 array (converged attractors)
 * @param ticks_out   Output: B int32 array (convergence ticks per frame)
 * @param states_out  Output: B int32 array (0=CONVERGED, 1=OSCILLATING, 2=CHAOTIC)
 * @param batch_size  Number of frames in the batch
 * @param max_iters   Maximum iterations before declaring CHAOTIC
 * @return 0 on success, -1 on error
 */
int ca_evolve_batch(const float* frames_in,
                    float* frames_out,
                    int* ticks_out,
                    int* states_out,
                    int batch_size,
                    int max_iters)
{
    size_t frame_bytes = (size_t)batch_size * GRID_SIZE * sizeof(float);
    size_t delta_bytes = (size_t)batch_size * GRID_SIZE * sizeof(float);
    size_t mean_bytes  = (size_t)batch_size * sizeof(float);

    float *d_buf_a   = nullptr;
    float *d_buf_b   = nullptr;
    float *d_deltas  = nullptr;
    float *d_means   = nullptr;
    bool  *converged = nullptr;
    float *h_means   = nullptr;
    int    retval    = 0;

    hipError_t err;

    /* Allocate device memory */
    err = hipMalloc(&d_buf_a, frame_bytes);
    if (err != hipSuccess) { fprintf(stderr, "hipMalloc A: %s\n", hipGetErrorString(err)); retval = -1; goto cleanup; }
    err = hipMalloc(&d_buf_b, frame_bytes);
    if (err != hipSuccess) { fprintf(stderr, "hipMalloc B: %s\n", hipGetErrorString(err)); retval = -1; goto cleanup; }
    err = hipMalloc(&d_deltas, delta_bytes);
    if (err != hipSuccess) { fprintf(stderr, "hipMalloc deltas: %s\n", hipGetErrorString(err)); retval = -1; goto cleanup; }
    err = hipMalloc(&d_means, mean_bytes);
    if (err != hipSuccess) { fprintf(stderr, "hipMalloc means: %s\n", hipGetErrorString(err)); retval = -1; goto cleanup; }

    /* Copy input to device */
    (void)hipMemcpy(d_buf_a, frames_in, frame_bytes, hipMemcpyHostToDevice);

    /* Initialize outputs */
    for (int b = 0; b < batch_size; b++) {
        ticks_out[b] = max_iters;
        states_out[b] = 2;  /* CHAOTIC until proven otherwise */
    }

    /* Track which frames have converged */
    converged = new bool[batch_size]();
    h_means   = new float[batch_size];

    {
        int total_cells = batch_size * GRID_SIZE;
        int block_size = 256;
        int grid_size = (total_cells + block_size - 1) / block_size;

        float* src = d_buf_a;
        float* dst = d_buf_b;

        for (int iter = 0; iter < max_iters; iter++) {
            /* Launch CA step */
            hipLaunchKernelGGL(ca_step_kernel,
                               dim3(grid_size), dim3(block_size), 0, 0,
                               src, dst, d_deltas, batch_size);

            /* Reduce deltas per frame */
            hipLaunchKernelGGL(reduce_delta_kernel,
                               dim3(batch_size), dim3(256), 0, 0,
                               d_deltas, d_means, batch_size);

            (void)hipDeviceSynchronize();

            /* Read back mean deltas */
            (void)hipMemcpy(h_means, d_means, mean_bytes, hipMemcpyDeviceToHost);

            /* Check convergence for each frame */
            bool all_done = true;
            for (int b = 0; b < batch_size; b++) {
                if (!converged[b]) {
                    if (h_means[b] < STABILITY_THRESHOLD) {
                        converged[b] = true;
                        ticks_out[b] = iter + 1;
                        states_out[b] = 0;  /* CONVERGED */
                    } else {
                        all_done = false;
                    }
                }
            }

            /* Swap buffers */
            float* tmp = src;
            src = dst;
            dst = tmp;

            if (all_done) break;
        }

        /* Copy final frames back (from src, which has the latest data) */
        (void)hipMemcpy(frames_out, src, frame_bytes, hipMemcpyDeviceToHost);
    }

cleanup:
    delete[] converged;
    delete[] h_means;
    if (d_buf_a)  (void)hipFree(d_buf_a);
    if (d_buf_b)  (void)hipFree(d_buf_b);
    if (d_deltas) (void)hipFree(d_deltas);
    if (d_means)  (void)hipFree(d_means);

    return retval;
}


/**
 * Convenience: evolve a single frame.
 */
int ca_evolve_single(const float* frame_in,
                     float* frame_out,
                     int* ticks_out,
                     int* state_out,
                     int max_iters)
{
    return ca_evolve_batch(frame_in, frame_out, ticks_out, state_out, 1, max_iters);
}

}  /* extern "C" */
