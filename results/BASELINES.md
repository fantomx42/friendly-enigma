# MMLU Benchmark Baselines

## Zero-Shot Cortex (no stored knowledge)
- **Date**: 2026-03-20
- **Mode**: `--mode cortex --all`
- **Score**: 24.3% (3,418/14,042)
- **Encoder**: blended (hippocampus 0.7 + language wheeler 0.3)
- **Stored memories**: 0 (science chunk wiped)
- **L3 classifier**: untrained (argmax on settled opinions)
- **Log**: `mmlu_cortex_zeroshot_2026-03-20.log`
- **Notes**: At chance (25% for 4-way MC). No knowledge to retrieve — cortex reasons purely over choice attractor geometry. Top: medical_genetics 32%, econometrics 31.6%. Bottom: international_law 15.7%.

## Cortex + Learned Facts (dev/val stored, no L3)
- **Date**: 2026-03-21
- **Mode**: `--mode cortex --all` (after `--mode learn`)
- **Score**: 25.3% (3,557/14,042)
- **Encoder**: blended (hippocampus 0.7 + language wheeler 0.3)
- **Stored memories**: 1,812 science attractors (MMLU dev+val correct Q&A)
- **L3 classifier**: untrained (argmax on settled opinions)
- **Log**: `mmlu_cortex_learned_2026-03-21.log`
- **Notes**: +1.0% over zero-shot. Top: formal_logic 36.5%, computer_security 32%, security_studies 31.8%. Knowledge retrieval helping some subjects but L3 classifier needed to fully leverage cortex features.

## Cortex + Learned Facts + L3 Classifier
- **Date**: 2026-03-21
- **Mode**: `--mode cortex --all --classifier-weights cortex_classifier.npz`
- **Score**: 25.9% (3,643/14,042)
- **Encoder**: blended (hippocampus 0.7 + language wheeler 0.3)
- **Stored memories**: 1,812 science attractors (MMLU dev+val)
- **L3 classifier**: trained 10 epochs on dev+val (11K params, numpy SGD)
- **Log**: `mmlu_cortex_l3_2026-03-21.log`
- **Notes**: +1.6% over zero-shot, +0.6% over learned-without-L3. Top: jurisprudence 36.1%, public_relations 34.5%, conceptual_physics 33.6%. L3 helps modestly — loss barely moved from chance (1.39 vs 1.386), needs more training data or richer features.

## Cortex + Trajectory Similarity (hybrid retrieval, no L3)
- **Date**: 2026-03-21
- **Mode**: `--mode cortex --all` (with trajectory cache)
- **Score**: 25.2% (3,545/14,042)
- **Encoder**: blended (hippocampus 0.7 + language wheeler 0.3)
- **Stored memories**: 3,207 (all chunks) with trajectory signatures
- **L3 classifier**: untrained (argmax on settled opinions)
- **Trajectory alpha**: 0.7 (attractor) / 0.3 (trajectory)
- **Log**: `mmlu_cortex_trajectory_2026-03-21.log`
- **Notes**: -0.1% vs learned-facts baseline. With L3: 25.9% (same as without trajectory). Trajectory re-ranking neutral — doesn't help or hurt. Hypotheses: (1) alpha=0.7 too conservative — trajectory signal drowned out, (2) hippocampus n-gram encoding produces similar seed frames for lexically similar text → similar trajectories, meaning trajectory doesn't add new discrimination power, (3) need to test lower alpha values to weight trajectory higher. Tunable: alpha, similarity weights, curve length.

## Previous MiniLM Semantic Baseline
- **Date**: 2026-03-17
- **Mode**: `--mode semantic` (MiniLM encoder)
- **Score**: 27.5%
- **Notes**: Used external pretrained model. Removed in cortex transition.
