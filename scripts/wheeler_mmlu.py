"""Wheeler Memory — MMLU Benchmark Harness

Evaluates Wheeler against the Massive Multitask Language Understanding benchmark
and reports accuracy per subject, comparable to frontier model scores.

Evaluation modes
----------------
semantic (default, Wheeler-native)
    For each choice A/B/C/D, query Wheeler with "question + choice" and measure
    the top attractor similarity. Pick the choice with the highest score.
    No LLM involved — the CA system picks the answer.

decode
    Recall memories for the question, then pass Wheeler's attractor state +
    the four choices to the small model decoder. Model outputs a single letter.

learn
    Full learn→consolidate→test cycle:
    1. Embed all correct Q&A pairs from dev+validation splits.
    2. Store each as a new attractor in the science chunk.
    3. Run sleep consolidation to reinforce and prune.
    4. Rebuild the attractor cache and test on the test split.

recall-text
    Recall top-K facts, extract answer by letter regex or text matching from context.

cortex
    Full cortex pipeline: L1 graph reasoning → L2 settlement → SCM evaluation
    → optional L3 neural network classifier. Retrieves memories into attractor
    graph, runs settlement CA for opinion diffusion, computes coherence layers,
    and picks answer with highest settled opinion (or via L3 classifier if weights provided).

Usage
-----
    python scripts/wheeler_mmlu.py --subjects abstract_algebra --samples 50
    python scripts/wheeler_mmlu.py --mode decode --model qwen2.5:1.5b --samples 20
    python scripts/wheeler_mmlu.py --mode learn --subjects high_school_physics
    python scripts/wheeler_mmlu.py --mode cortex --subjects abstract_algebra --recall-k 10
    python scripts/wheeler_mmlu.py --mode cortex --classifier-weights path/to/weights.npz --all
    python scripts/wheeler_mmlu.py --all --samples 10
    python scripts/wheeler_mmlu.py --list-subjects
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from wheeler_memory.storage import recall_memory

# ---------------------------------------------------------------------------
# MMLU subject catalogue (57 subjects)
# ---------------------------------------------------------------------------

MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology",
    "us_foreign_policy", "virology", "world_religions",
]

CHOICES = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# Attractor cache — loaded once at benchmark startup for fast Pearson search
# ---------------------------------------------------------------------------

class AttractorCache:
    """Pre-loads all stored attractors into a single numpy matrix.

    Replaces per-recall disk reads with one vectorized matrix operation.
    For 1428 attractors: disk load ~0.5s once, then <1ms per query.
    """

    def __init__(self, data_dir: Path | None = None):
        from wheeler_memory.storage import DEFAULT_DATA_DIR, list_memories
        from wheeler_memory.dynamics import evolve_and_interpret

        d = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        print("  Loading attractor cache into memory...")
        t0 = time.time()

        memories = list_memories(data_dir=d)
        self._texts: list[str] = []
        self._meta: list[dict] = []
        attractor_list = []

        for m in memories:
            chunk = m.get("chunk", "general")
            hex_key = m["hex_key"]
            att_path = d / "chunks" / chunk / "attractors" / f"{hex_key}.npy"
            if not att_path.exists():
                continue
            att = np.load(att_path)
            if att.shape != (64, 64):
                continue
            attractor_list.append(att.flatten().astype(np.float32))
            self._texts.append(m["text"])
            self._meta.append(m)

        if not attractor_list:
            self._matrix = np.zeros((0, 4096), dtype=np.float32)
            self._att_means = np.zeros(0, dtype=np.float32)
            self._att_stds = np.zeros(0, dtype=np.float32)
        else:
            self._matrix = np.stack(attractor_list)           # (N, 4096)
            self._att_means = self._matrix.mean(axis=1)       # (N,)
            self._att_stds = self._matrix.std(axis=1)         # (N,)

        self._evolve = evolve_and_interpret
        elapsed = time.time() - t0
        print(f"  Cached {len(self._texts)} attractors in {elapsed:.2f}s")

    def search(self, query_frame: np.ndarray, top_k: int = 5) -> list[dict]:
        """Vectorized Pearson search against entire attractor matrix."""
        if len(self._texts) == 0:
            return []

        from wheeler_memory.dynamics import evolve_and_interpret
        from wheeler_memory.constants import SALIENCE_MAX_ITERS_MED, SALIENCE_THRESHOLD_MED

        result = evolve_and_interpret(
            query_frame,
            max_iters=SALIENCE_MAX_ITERS_MED,
            stability_threshold=SALIENCE_THRESHOLD_MED,
        )
        q_flat = result["attractor"].flatten().astype(np.float32)
        q_mean = q_flat.mean()
        q_std = q_flat.std()

        if q_std < 1e-10:
            return []

        # Vectorized Pearson: (N, 4096) dot (4096,) → (N,)
        q_centered = q_flat - q_mean
        centered = self._matrix - self._att_means[:, None]
        dots = centered @ q_centered                                      # (N,)
        valid = self._att_stds > 1e-10
        sims = np.where(valid, dots / (4096 * self._att_stds * q_std), 0.0)

        top_idx = np.argpartition(sims, -min(top_k, len(sims)))[-top_k:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

        return [
            {"text": self._texts[i], "similarity": float(sims[i])}
            for i in top_idx
        ]


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_mmlu(subjects: list[str], split: str = "test", samples: int | None = None):
    """Yield (subject, question, choices, answer_idx) tuples."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library required. Run: pip install datasets", file=sys.stderr)
        sys.exit(1)

    for subject in subjects:
        try:
            ds = load_dataset("cais/mmlu", subject, split=split)
        except Exception as exc:
            print(f"  [skip] {subject}: {exc}", file=sys.stderr)
            continue

        if samples:
            ds = ds.select(range(min(samples, len(ds))))

        for item in ds:
            choices = [item["choices"][i] for i in range(4)]
            answer_idx = int(item["answer"])
            yield subject, item["question"], choices, answer_idx


# ---------------------------------------------------------------------------
# Semantic scoring (Wheeler-native, no LLM)
# ---------------------------------------------------------------------------

def score_semantic(
    question: str,
    choices: list[str],
    recall_k: int = 5,
    cache: "AttractorCache | None" = None,
    precomputed_frames: list | None = None,
) -> tuple[int, list[float]]:
    """Score choices by Pearson correlation against the attractor cache.

    Uses vectorized matrix search (AttractorCache) for speed — avoids all
    per-recall disk I/O. Falls back to hash_to_frame if no cache provided.

    Returns (predicted_index, [score_A, score_B, score_C, score_D]).
    """
    from wheeler_memory.hashing import hash_to_frame

    if precomputed_frames is not None:
        frames = precomputed_frames
    else:
        frames = [hash_to_frame(f"{question} {c}") for c in choices]

    scores = []
    for frame in frames:
        if cache is not None:
            hits = cache.search(frame, top_k=recall_k)
        else:
            hits = []
        top_sim = max((h.get("similarity", 0.0) for h in hits), default=0.0)
        scores.append(top_sim)
    return int(scores.index(max(scores))), scores


def score_recall_text(
    question: str,
    choices: list[str],
    recall_k: int = 10,
    cache: "AttractorCache | None" = None,
    debug: bool = False,
) -> tuple[int, float]:
    """Recall top-K facts; extract answer by letter regex OR text matching.

    Two extraction strategies (both tried per hit):
      1. Letter regex: "A: B." → vote for choice index 1
      2. Text match:  answer substring appears in a choice → vote for that choice

    Returns (predicted_index, confidence) or (-1, 0.0) on failure.
    """
    try:
        from wheeler_memory.hippocampus import hippocampus_to_frame
        frame = hippocampus_to_frame(question).flatten()
    except Exception:
        from wheeler_memory.hashing import hash_to_frame
        frame = hash_to_frame(question)

    hits = cache.search(frame, top_k=recall_k) if cache is not None else []

    if debug:
        print(f"      [debug] top-{len(hits)} recalls:")
        for h in hits[:5]:
            print(f"        sim={h['similarity']:.3f}  {h['text'][:80]}")

    votes = [0.0, 0.0, 0.0, 0.0]  # indexed by choice position
    choices_lower = [c.lower().strip() for c in choices]

    for hit in hits:
        text = hit.get("text", "")
        sim = hit.get("similarity", 0.0)
        matched = False

        # Strategy 1: letter regex (MMLU learn-mode format "A: B. choice_text")
        m = re.search(r"A:\s*([A-D])\.", text)
        if m:
            idx = CHOICES.index(m.group(1))
            votes[idx] += sim
            matched = True

        # Strategy 2: text match (SciQ/ARC format "A: answer_text")
        if not matched:
            # Extract answer portion after "A:" or "A: "
            ans_m = re.search(r"A:\s*(.+?)(?:\.|$)", text)
            if ans_m:
                answer = ans_m.group(1).lower().strip()
                if len(answer) >= 2:  # skip trivially short
                    for ci, cl in enumerate(choices_lower):
                        if answer in cl or cl in answer:
                            votes[ci] += sim
                            matched = True
                            break

    best_idx = int(max(range(4), key=lambda i: votes[i]))
    if votes[best_idx] == 0.0:
        return -1, 0.0
    return best_idx, votes[best_idx]


def score_cortex(
    question: str,
    choices: list[str],
    recall_k: int = 10,
    cache: "AttractorCache | None" = None,
    classifier_weights=None,  # ClassifierWeights | None
) -> tuple[int, float]:
    """Score using full cortex pipeline: L1 graph → L2 settlement → SCM → optional L3.

    Returns (predicted_index, confidence).
    """
    from wheeler_memory.hippocampus import hippocampus_to_frame
    from wheeler_memory.dynamics import evolve_and_interpret
    from wheeler_memory.cortex import cortex_reason
    from wheeler_memory.cortex_scm import compute_scm

    # 1. Encode question → frame → attractor
    q_frame = hippocampus_to_frame(question)
    q_result = evolve_and_interpret(q_frame)
    q_attractor = q_result["attractor"].flatten().astype(np.float32)

    # 2. Encode each choice → attractor
    choice_attractors = []
    for c in choices:
        c_frame = hippocampus_to_frame(f"{question} {c}")
        c_result = evolve_and_interpret(c_frame)
        choice_attractors.append(c_result["attractor"].flatten().astype(np.float32))
    choice_attractors = np.array(choice_attractors)  # (4, 4096)

    # 3. Retrieve top-K from cache (if available)
    retrieved_attractors = []
    retrieved_sims = []
    if cache is not None:
        hits = cache.search(q_frame, top_k=recall_k)
        for h in hits:
            # We need the actual attractor, not just text
            # Re-encode from text (cache stores text + similarity)
            r_frame = hippocampus_to_frame(h["text"])
            r_result = evolve_and_interpret(r_frame)
            retrieved_attractors.append(r_result["attractor"].flatten().astype(np.float32))
            retrieved_sims.append(h["similarity"])

    # 4. Build attractor ensemble: retrieved + choices
    all_attractors = []
    all_sims = []
    if retrieved_attractors:
        all_attractors.extend(retrieved_attractors)
        all_sims.extend(retrieved_sims)
    # Always include the 4 choice attractors
    for i, ca in enumerate(choice_attractors):
        all_attractors.append(ca)
        # Choice similarity = Pearson between query and choice attractor
        q_centered = q_attractor - q_attractor.mean()
        c_centered = ca - ca.mean()
        q_std = q_centered.std()
        c_std = c_centered.std()
        if q_std > 1e-10 and c_std > 1e-10:
            sim = float(np.dot(q_centered, c_centered) / (len(q_centered) * q_std * c_std))
        else:
            sim = 0.0
        all_sims.append(sim)

    all_attractors = np.array(all_attractors)
    all_sims = np.array(all_sims)

    # 5. Run cortex
    cortex_result = cortex_reason(all_attractors, all_sims)
    graph = cortex_result["graph"]
    settlement = cortex_result["settlement"]

    # 6. Extract choice similarities from settled opinions
    n_retrieved = len(retrieved_attractors)
    choice_settled = settlement["settled"][-4:]  # last 4 entries are choices

    # 7. If we have L3 classifier weights, use them
    if classifier_weights is not None:
        from wheeler_memory.cortex_classifier import classify
        choice_sims = all_sims[-4:]  # Get the 4 choice similarities
        scm_result = compute_scm(
            np.array(all_sims[:n_retrieved]) if n_retrieved > 0 else np.array([]),
            graph.adjacency,
            graph.cluster_labels,
            settlement["settled"],
            settlement["settled"],  # prev = initial for first pass
            float(choice_sims.max()) if len(choice_sims) > 0 else 0.0,
        )
        scm_layers = np.array([
            scm_result.temperature,
            scm_result.salience,
            scm_result.energy,
            scm_result.integration,
            scm_result.polarity,
            scm_result.net_warrant,
            scm_result.explanation_readiness,
        ])
        pred_idx, confidence = classify(settlement["settled"], choice_sims, scm_layers, classifier_weights)
        return pred_idx, confidence

    # 8. Without L3: pick choice with highest settled opinion
    pred_idx = int(np.argmax(choice_settled))
    confidence = float(choice_settled[pred_idx])
    return pred_idx, confidence


def precompute_all_frames(questions_and_choices: list[tuple[str, list[str]]]) -> list[list]:
    """Batch-encode all question+choice pairs in one encoder call.

    Returns list of [frame_A, frame_B, frame_C, frame_D] per question.
    Much faster than encoding 4 texts per question individually.
    """
    from wheeler_memory.hippocampus import hippocampus_to_frame_batch

    all_queries = []
    for question, choices in questions_and_choices:
        all_queries.extend(f"{question} {c}" for c in choices)

    print(f"  Pre-encoding {len(all_queries)} query+choice pairs in one batch...")
    all_frames_2d = hippocampus_to_frame_batch(all_queries)
    all_frames = [f.flatten() for f in all_frames_2d]

    result = []
    for i in range(0, len(all_frames), 4):
        result.append(list(all_frames[i:i + 4]))
    return result


# ---------------------------------------------------------------------------
# Decode scoring (wheeler-primary decoder picks A/B/C/D)
# ---------------------------------------------------------------------------

_MMLU_SYSTEM_PROMPT = """\
You are a language renderer for Wheeler Memory.

Your ONLY job is to express the memory state below as a single-letter answer.
Do NOT add knowledge from your own training data. Do NOT speculate.

You must reply with exactly one letter: A, B, C, or D — nothing else.
"""


def _ollama_chat(messages: list[dict], model: str, base_url: str) -> str:
    payload = {"model": model, "messages": messages, "stream": False}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read()).get("message", {}).get("content", "")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama unreachable at {base_url}: {exc}") from exc


def score_decode(
    question: str,
    choices: list[str],
    model: str,
    ollama_url: str,
    recall_k: int = 5,
    data_dir: Path | None = None,
) -> tuple[int, float]:
    """Wheeler recalls, decoder picks A/B/C/D. Returns (predicted_index, confidence)."""
    from wheeler_memory.decoder import extract_state, format_state

    hits = recall_memory(
        question,
        top_k=recall_k,
        data_dir=data_dir,
        encoder="blended",
    )
    state = extract_state(question, hits)
    memory_block = format_state(state)

    choice_block = "\n".join(f"{CHOICES[i]}. {c}" for i, c in enumerate(choices))
    user_content = (
        f"{memory_block}\n\n"
        f"QUESTION: {question}\n\n"
        f"OPTIONS:\n{choice_block}\n\n"
        "Reply with exactly one letter (A, B, C, or D)."
    )

    messages = [
        {"role": "system", "content": _MMLU_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    raw = _ollama_chat(messages, model, ollama_url).strip().upper()
    confidence = state.confidence

    for letter in CHOICES:
        if raw.startswith(letter):
            return CHOICES.index(letter), confidence

    return -1, confidence  # parse failure


# ---------------------------------------------------------------------------
# Learn mode helpers
# ---------------------------------------------------------------------------

def store_learned_fact(text: str, frame: np.ndarray, data_dir=None, chunk: str = 'science') -> str:
    """Embed → evolve → brick → store. Returns hex_key."""
    from wheeler_memory.dynamics import evolve_and_interpret
    from wheeler_memory.brick import MemoryBrick
    from wheeler_memory.storage import store_memory
    result = evolve_and_interpret(frame)
    brick = MemoryBrick.from_evolution_result(result)
    return store_memory(text, result, brick, data_dir=data_dir, chunk=chunk, auto_evict=False)


def run_learning_pass(subjects: list[str], data_dir=None):
    """Embed all correct Q&A pairs from dev+validation, store each, then consolidate.

    Returns (n_stored, consolidation_result).
    """
    from wheeler_memory.hippocampus import hippocampus_to_frame_batch
    from wheeler_memory.consolidation import sleep_consolidate
    from wheeler_memory.eviction import evict_for_capacity

    all_texts = []
    choices_letters = ['A', 'B', 'C', 'D']

    for subject in subjects:
        for split in ['dev', 'validation']:
            items = list(load_mmlu([subject], split=split))
            for _, question, choices, answer_idx in items:
                correct_letter = choices_letters[answer_idx]
                correct_text = choices[answer_idx]
                all_texts.append(f"Q: {question} A: {correct_letter}. {correct_text}")

    if not all_texts:
        return 0, None

    print(f"\n  [learn] Encoding {len(all_texts)} correct Q&A pairs...")
    frames = hippocampus_to_frame_batch(all_texts)

    stored = 0
    for text, frame in zip(all_texts, frames):
        store_learned_fact(text, frame, data_dir=data_dir, chunk='science')
        stored += 1
    print(f"  [learn] Stored {stored} facts.")

    d = Path(data_dir) if data_dir else Path.home() / '.wheeler_memory'

    print("  [learn] Running eviction to enforce capacity limits...")
    evict_for_capacity(d)

    print("  [learn] Running sleep consolidation on science chunk...")
    result = sleep_consolidate(d, chunk='science')
    n_consolidated = len(result.memories_consolidated)
    print(f"  [learn] Consolidation: {n_consolidated} memories consolidated, "
          f"{result.total_frames_before} → {result.total_frames_after} frames")

    return stored, result


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    subjects: list[str],
    mode: str,
    samples: int | None,
    recall_k: int,
    model: str,
    ollama_url: str,
    data_dir: Path | None,
    output: Path | None,
    split: str,
    use_embed: bool = True,
    debug_recall: bool = False,
    classifier_weights_path: str | None = None,
) -> dict:
    results = {}          # subject → {correct, total, accuracy}
    all_rows = []         # for TSV output

    total_correct = 0
    total_questions = 0

    embed_label = "embed" if use_embed else "hash"
    is_learn_mode = mode == 'learn'
    print(f"\n{'='*70}")
    print(f"  WHEELER MMLU BENCHMARK")
    if is_learn_mode:
        print(f"  Mode: LEARN → SEMANTIC   Subjects: {len(subjects)}   Test split: {split}")
        print(f"  Phase 1: learn dev+validation → consolidate")
        print(f"  Phase 2: semantic test on {split} split")
    else:
        print(f"  Mode: {mode.upper()}({embed_label})   Subjects: {len(subjects)}   Split: {split}")
    if samples:
        print(f"  Samples per subject: {samples}")
    print(f"{'='*70}")

    # Learn mode: store dev+validation facts, consolidate, then test on test split
    if is_learn_mode:
        n_stored, cons = run_learning_pass(subjects, data_dir)
        # Rebuild attractor cache with newly stored memories
        print(f"\n  Rebuilding attractor cache after learning pass...")
        mode = 'semantic'

    print()

    # Load classifier weights if provided (for cortex L3)
    classifier_weights = None
    if classifier_weights_path and mode == "cortex":
        from wheeler_memory.cortex_classifier import load_weights
        print(f"  Loading classifier weights from {classifier_weights_path}...")
        classifier_weights = load_weights(classifier_weights_path)

    # Build attractor cache once for the entire benchmark run
    cache = AttractorCache(data_dir) if mode in ("semantic", "recall-text", "cortex") else None

    for subj in subjects:
        correct = 0
        total = 0
        parse_failures = 0

        print(f"  [{subj}]")

        # Load all questions for this subject upfront so we can pre-encode
        subject_items = list(load_mmlu([subj], split, samples))

        # Pre-encode all question+choice pairs in one batch (semantic mode only)
        precomputed = None
        if mode == "semantic" and use_embed and subject_items:
            qc_pairs = [(q, ch) for _, q, ch, _ in subject_items]
            precomputed = precompute_all_frames(qc_pairs)

        for idx, (subject, question, choices, answer_idx) in enumerate(subject_items):
            t0 = time.time()

            if mode == "semantic":
                frames = precomputed[idx] if precomputed is not None else None
                pred_idx, scores = score_semantic(question, choices, recall_k, cache, frames)
                confidence = max(scores)
            elif mode == "recall-text":
                _dbg = debug_recall and idx < 3
                pred_idx, confidence = score_recall_text(question, choices, recall_k, cache, debug=_dbg)
            elif mode == "cortex":
                pred_idx, confidence = score_cortex(question, choices, recall_k, cache, classifier_weights)
            else:
                try:
                    pred_idx, confidence = score_decode(
                        question, choices, model, ollama_url, recall_k, data_dir
                    )
                except RuntimeError as exc:
                    print(f"    ERROR: {exc}", file=sys.stderr)
                    sys.exit(1)

            elapsed = time.time() - t0
            is_correct = pred_idx == answer_idx
            if pred_idx == -1:
                parse_failures += 1

            if is_correct:
                correct += 1
            total += 1

            marker = "✓" if is_correct else "✗"
            pred_letter = CHOICES[pred_idx] if 0 <= pred_idx <= 3 else "?"
            true_letter = CHOICES[answer_idx]
            print(
                f"    {marker} pred={pred_letter} true={true_letter} "
                f"conf={confidence:.3f} ({elapsed:.1f}s) "
                f"{question[:55]}..."
            )

            all_rows.append({
                "subject": subject,
                "question": question[:100],
                "true": true_letter,
                "pred": pred_letter,
                "correct": is_correct,
                "confidence": round(confidence, 4),
                "elapsed_s": round(elapsed, 2),
            })

        accuracy = correct / total if total else 0.0
        results[subj] = {"correct": correct, "total": total, "accuracy": accuracy}
        total_correct += correct
        total_questions += total

        if parse_failures:
            print(f"    → {correct}/{total} correct ({accuracy:.1%})  [{parse_failures} parse failures]")
        else:
            print(f"    → {correct}/{total} correct ({accuracy:.1%})")

    overall = total_correct / total_questions if total_questions else 0.0

    print(f"\n{'='*70}")
    print(f"  OVERALL: {total_correct}/{total_questions} = {overall:.1%}")
    print(f"{'='*70}")
    print(f"\n  Per-subject accuracy:")
    for subj, r in sorted(results.items(), key=lambda x: -x[1]["accuracy"]):
        bar = "█" * int(r["accuracy"] * 20)
        print(f"    {subj:<40} {r['accuracy']:5.1%}  {bar}")

    # Save results
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()) if all_rows else [])
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n  Results saved to: {output}")

    return {"subjects": results, "overall": overall, "total": total_questions}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="wheeler-mmlu",
        description="Run Wheeler Memory against the MMLU benchmark.",
    )
    p.add_argument(
        "--subjects", "-s",
        nargs="+",
        metavar="SUBJECT",
        help="MMLU subject name(s) to evaluate.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Run all 57 MMLU subjects.",
    )
    p.add_argument(
        "--list-subjects",
        action="store_true",
        help="Print all available subjects and exit.",
    )
    p.add_argument(
        "--samples", "-n",
        type=int,
        default=None,
        metavar="N",
        help="Max questions per subject (default: all).",
    )
    p.add_argument(
        "--mode",
        choices=["semantic", "decode", "learn", "recall-text", "cortex"],
        default="semantic",
        help="Scoring mode: 'semantic' (Wheeler-native), 'decode' (small model), 'learn' (store dev+val → consolidate → test), 'recall-text' (extract answer letter from recalled facts), or 'cortex' (full cortex pipeline). Default: semantic.",
    )
    p.add_argument(
        "--split",
        choices=["test", "validation", "dev"],
        default="test",
        help="Dataset split to use (default: test).",
    )
    p.add_argument(
        "--recall-k",
        type=int,
        default=5,
        metavar="K",
        help="Memories to recall per query (default: 5).",
    )
    p.add_argument(
        "--model",
        default="qwen2.5:1.5b",
        metavar="MODEL",
        help="Ollama model for decode mode (default: qwen2.5:1.5b).",
    )
    p.add_argument(
        "--ollama",
        default="http://localhost:11434",
        metavar="URL",
        help="Ollama base URL (default: http://localhost:11434).",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        metavar="DIR",
        help="Wheeler Memory data directory.",
    )
    p.add_argument(
        "--output", "-o",
        default=None,
        metavar="FILE",
        help="Save per-question results to this TSV/CSV file.",
    )
    p.add_argument(
        "--no-embed",
        action="store_true",
        help="Use hash-based frames instead of sentence embeddings (much faster, lower accuracy).",
    )
    p.add_argument(
        "--debug-recall",
        action="store_true",
        help="Print top-5 recalled texts for first 3 questions per subject (recall-text mode).",
    )
    p.add_argument(
        "--classifier-weights",
        type=str,
        default=None,
        metavar="FILE",
        help="Path to trained cortex classifier weights (.npz) for cortex mode L3 classification.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.list_subjects:
        print("\n".join(MMLU_SUBJECTS))
        return

    if args.all:
        subjects = MMLU_SUBJECTS
    elif args.subjects:
        subjects = args.subjects
    else:
        print(
            "ERROR: Specify subjects with --subjects or use --all.\n"
            "       Run --list-subjects to see all options.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate subjects
    unknown = [s for s in subjects if s not in MMLU_SUBJECTS]
    if unknown:
        print(f"ERROR: Unknown subjects: {unknown}", file=sys.stderr)
        sys.exit(1)

    data_dir = Path(args.data_dir) if args.data_dir else None

    run_benchmark(
        subjects=subjects,
        mode=args.mode,
        samples=args.samples,
        recall_k=args.recall_k,
        model=args.model,
        ollama_url=args.ollama,
        data_dir=data_dir,
        output=args.output,
        split=args.split,
        use_embed=not args.no_embed,
        debug_recall=args.debug_recall,
        classifier_weights_path=args.classifier_weights,
    )


if __name__ == "__main__":
    main()
