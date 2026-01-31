# Phase 1 Findings: Literature & Dependency Review

## 1. Architecture & Requirements
**BitNet b1.58** is a 1-bit LLM variant (ternary weights: -1, 0, 1) that matches full-precision Transformer performance while being significantly more efficient.

*   **Key Constraint:** Models must be *trained* as BitNet from scratch. You cannot simply quantize an existing Llama-3 or Mistral model to 1.58-bit and expect it to work.
*   **Hardware:** Runs efficiently on CPU due to low memory bandwidth requirements, but also supports GPU acceleration.

## 2. Software Compatibility
*   **llama.cpp:** Full support exists. The architecture (RMSNorm, SwiGLU, Rotary Embeddings) maps well.
*   **Ollama:** Supported via GGUF. Since Ollama is built on `llama.cpp`, it can load `bitnet-b1.58` GGUF files directly.
*   **bitnet.cpp:** Microsoft's specialized inference engine offers further optimizations (kernels) but is not strictly required if `llama.cpp` performance is acceptable.

## 3. Available Models
The primary candidate for testing is:
*   **Model:** `microsoft/bitnet-b1.58-2B-4T`
*   **Format:** GGUF (`microsoft/bitnet-b1.58-2B-4T-gguf`)
*   **Size:** ~1-2GB (estimated for 2B params @ 1.58 bits, effectively ~2 bits per parameter with overhead).

## 4. Viability Assessment (Preliminary)
**High.** The existence of GGUF support means we can integrate this into the Ralph swarm (Tier 3 ASICs) using our standard `Ollama` backend without custom compilation or new infrastructure.

## Next Steps
Proceed to Phase 2:
1.  Download `bitnet-b1.58-2B-4T-gguf`.
2.  Import into Ollama (`ollama create bitnet-2b -f Modelfile`).
3.  Benchmark against `tinyllama` (1.1B) or `qwen2.5-coder:1.5b`.
