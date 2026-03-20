#!/usr/bin/env python3
"""Prepare a multi-domain corpus for Wheeler Memory crystallization.

Extracts text from available datasets and includes curated domain entries
for the semantic apple test.

Sources:
  - SWE-bench_Verified (500 entries, code domain)
  - mbpp sanitized (427 entries, code domain)
  - LongBench QA tasks (multi-domain)
  - Curated ML/AI concepts (~50 entries)
  - Curated science concepts (~50 entries)

Output: datasets/corpus.jsonl
"""

import json
import zipfile
from pathlib import Path

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"
OUTPUT = DATASETS_DIR / "corpus.jsonl"

MIN_LEN = 20
MAX_LEN = 5000


def _write_entries(out, texts, source):
    """Write texts to JSONL, filtering by length. Returns count written."""
    count = 0
    for text in texts:
        text = text.strip()
        if MIN_LEN <= len(text) <= MAX_LEN:
            out.write(json.dumps({"text": text, "source": source}) + "\n")
            count += 1
    return count


def extract_swe_bench(out) -> int:
    """Extract problem_statement from SWE-bench_Verified."""
    import pandas as pd

    path = DATASETS_DIR / "SWE-bench_Verified" / "data"
    if not path.exists():
        print("  SWE-bench not found, skipping")
        return 0

    df = pd.read_parquet(path)
    texts = df["problem_statement"].dropna().tolist()
    count = _write_entries(out, texts, "swe-bench")
    print(f"  SWE-bench: {count} entries")
    return count


def extract_mbpp(out) -> int:
    """Extract prompt from mbpp sanitized."""
    import pandas as pd

    path = DATASETS_DIR / "mbpp" / "sanitized"
    if not path.exists():
        print("  mbpp not found, skipping")
        return 0

    df = pd.read_parquet(path)
    texts = df["prompt"].dropna().tolist()
    count = _write_entries(out, texts, "mbpp")
    print(f"  mbpp: {count} entries")
    return count


def extract_longbench(out) -> int:
    """Extract input (question) from LongBench non-code tasks."""
    zip_path = DATASETS_DIR / "LongBench" / "data.zip"
    if not zip_path.exists():
        print("  LongBench not found, skipping")
        return 0

    # Non-code, non-Chinese tasks with meaningful questions
    include_tasks = {
        "hotpotqa", "2wikimqa", "musique", "narrativeqa", "qasper",
        "gov_report", "multi_news", "triviaqa", "trec", "samsum",
        "multifieldqa_en",
    }

    count = 0
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.endswith(".jsonl"):
                continue
            task = Path(name).stem
            # Skip _e (extended) variants
            if task.endswith("_e"):
                continue
            if task not in include_tasks:
                continue

            with z.open(name) as f:
                for line in f:
                    obj = json.loads(line.decode())
                    text = obj.get("input", "").strip()
                    if MIN_LEN <= len(text) <= MAX_LEN:
                        out.write(json.dumps({"text": text, "source": f"longbench/{task}"}) + "\n")
                        count += 1

    print(f"  LongBench: {count} entries")
    return count


def write_curated_ml(out) -> int:
    """Curated ML/AI concepts for apple test semantic density."""
    entries = [
        # Core ML concepts
        "Neural networks learn by adjusting weights through backpropagation of error gradients",
        "Gradient descent minimizes loss by iteratively updating parameters in the direction of steepest descent",
        "Stochastic gradient descent uses random minibatches to approximate the full gradient",
        "Learning rate controls the step size during gradient descent optimization",
        "Overfitting occurs when a model memorizes training data instead of learning general patterns",
        "Regularization techniques like dropout and weight decay prevent overfitting",
        "Batch normalization stabilizes training by normalizing layer inputs across minibatches",
        "Activation functions like ReLU introduce nonlinearity into neural network computations",

        # Attention and transformers
        "Attention mechanism allows models to focus on relevant parts of the input sequence",
        "Self-attention computes relationships between all positions in a sequence simultaneously",
        "Multi-head attention runs multiple attention operations in parallel for richer representations",
        "Scaled dot-product attention divides by square root of key dimension for stable gradients",
        "Positional encoding injects sequence order information since attention is permutation invariant",
        "Layer normalization normalizes activations across features within each training example",
        "Residual connections allow gradients to flow directly through the network via skip connections",
        "Feed-forward networks in transformers apply two linear transformations with a ReLU activation",

        # Architectures
        "Encoder-decoder architecture maps input sequences to output sequences through a bottleneck",
        "BERT uses bidirectional masked language modeling to pretrain deep representations",
        "GPT generates text autoregressively by predicting the next token given previous context",
        "Recurrent neural networks maintain hidden state to process sequential data one step at a time",
        "Long short-term memory networks use gates to selectively remember and forget information",
        "Convolutional neural networks extract hierarchical spatial features using learned filters",
        "Generative adversarial networks pit a generator against a discriminator in a minimax game",
        "Variational autoencoders learn latent representations by maximizing a variational lower bound",
        "Diffusion models generate data by learning to reverse a gradual noising process",

        # Training and optimization
        "Cross-entropy loss measures the divergence between predicted and true probability distributions",
        "Adam optimizer combines momentum and adaptive learning rates for efficient training",
        "Transfer learning reuses pretrained model weights as initialization for new tasks",
        "Fine-tuning adapts a pretrained model to a specific task by training on task-specific data",
        "Data augmentation increases training set diversity through random transformations",
        "Early stopping halts training when validation performance stops improving to prevent overfitting",
        "Curriculum learning trains on easy examples first then gradually increases difficulty",

        # Evaluation
        "Precision measures the fraction of positive predictions that are actually correct",
        "Recall measures the fraction of actual positives that are correctly identified",
        "F1 score is the harmonic mean of precision and recall balancing both metrics",
        "Perplexity measures how well a language model predicts a held-out text corpus",
        "BLEU score evaluates machine translation quality by comparing n-gram overlap with references",

        # Bridge sentences — connective tissue between concepts
        "The transformer architecture is built around self-attention as its core mechanism allowing it to weigh relationships between all sequence positions simultaneously",
        "BERT and GPT both use the transformer architecture for pretraining but BERT uses bidirectional masked prediction while GPT uses autoregressive left-to-right generation",
        "The encoder-decoder architecture also called sequence-to-sequence maps variable-length inputs to variable-length outputs through an intermediate representation",
        "Transformers replaced recurrent neural networks by using self-attention instead of sequential hidden state updates for parallel processing of sequences",
        "Multi-head attention in transformers projects queries keys and values into multiple subspaces enabling the model to attend to different relationship types simultaneously",

        # Specific techniques
        "Word embeddings map discrete tokens to dense continuous vectors preserving semantic similarity",
        "Tokenization breaks text into subword units like byte-pair encoding for open vocabulary models",
        "Knowledge distillation trains a small student model to mimic a larger teacher model",
        "Mixture of experts routes different inputs to specialized subnetworks for efficient scaling",
        "Reinforcement learning from human feedback aligns language models with human preferences",
        "Chain of thought prompting elicits step-by-step reasoning from large language models",
        "Retrieval augmented generation grounds language model responses in retrieved documents",
        "Quantization reduces model precision from 32-bit to 8-bit or 4-bit for efficient inference",
        "Low-rank adaptation fine-tunes models by adding small trainable matrices to frozen weights",
    ]
    count = _write_entries(out, entries, "curated/ml")
    print(f"  Curated ML/AI: {count} entries")
    return count


def write_curated_science(out) -> int:
    """Curated science concepts for apple test semantic density."""
    entries = [
        # Physics
        "Newton's first law states objects remain at rest or in uniform motion unless acted upon by a force",
        "Newton's second law relates force to mass times acceleration F equals ma",
        "Newton's third law says every action has an equal and opposite reaction",
        "Conservation of energy states energy cannot be created or destroyed only transformed",
        "Conservation of momentum means total momentum in a closed system remains constant",
        "Einstein's special relativity shows time dilates and length contracts at high velocities",
        "Einstein's general relativity describes gravity as curvature of spacetime by mass and energy",
        "Electromagnetic waves propagate at the speed of light through oscillating electric and magnetic fields",
        "The photoelectric effect demonstrates that light behaves as discrete packets called photons",
        "Wave-particle duality means quantum objects exhibit both wave and particle behavior",
        "Heisenberg's uncertainty principle limits simultaneous knowledge of position and momentum",
        "Quantum superposition allows particles to exist in multiple states until measured",
        "Schrodinger's equation describes how quantum states evolve over time",
        "The standard model classifies all known fundamental particles and three of four forces",
        "Dark matter interacts gravitationally but does not emit absorb or reflect light",
        "Dark energy drives the accelerating expansion of the universe",
        "Thermodynamics governs heat energy transfer and the direction of spontaneous processes",
        "Entropy measures the disorder or number of microscopic configurations of a system",

        # Chemistry
        "Chemical bonds form when atoms share or transfer electrons to achieve stable configurations",
        "Covalent bonds share electron pairs between atoms while ionic bonds transfer electrons",
        "The periodic table organizes elements by atomic number and chemical properties",
        "Catalysts speed up chemical reactions without being consumed in the process",
        "pH measures the concentration of hydrogen ions in solution on a logarithmic scale",
        "Oxidation-reduction reactions involve the transfer of electrons between species",
        "Organic chemistry studies carbon-based compounds and their reactions",
        "Polymers are large molecules made of repeating structural units called monomers",

        # Biology
        "DNA encodes genetic information using four nucleotide bases adenine thymine guanine cytosine",
        "RNA transcription copies DNA sequences into messenger RNA for protein synthesis",
        "Ribosomes translate mRNA codons into amino acid sequences to build proteins",
        "Mitochondria generate ATP through oxidative phosphorylation the cell's energy currency",
        "Photosynthesis converts carbon dioxide and water into glucose using sunlight energy",
        "Natural selection drives evolution by favoring traits that improve reproductive fitness",
        "Mutations introduce genetic variation that provides raw material for evolution",
        "Cell division through mitosis produces identical copies for growth and repair",
        "Meiosis produces gametes with half the chromosome number enabling sexual reproduction",
        "The immune system defends against pathogens using innate and adaptive responses",
        "Enzymes are biological catalysts that accelerate specific biochemical reactions",
        "Homeostasis maintains stable internal conditions despite changing external environments",

        # Bridge sentences — connective tissue between concepts
        "Newton's laws of motion and the conservation of energy are foundational principles of classical mechanics where force equals mass times acceleration and energy transforms between kinetic and potential forms",
        "Photosynthesis in chloroplasts captures sunlight to produce glucose while mitochondria break down glucose through oxidative phosphorylation to produce ATP completing the cell's energy cycle",
        "The energy flow in cells connects photosynthesis which stores solar energy in glucose to mitochondrial respiration which releases that energy as ATP for cellular work",
        "The immune system relies on enzymes such as lysozyme and proteases to break down pathogen components as part of both innate and adaptive defense mechanisms",

        # Earth science / astronomy
        "Plate tectonics describes the movement of lithospheric plates on the asthenosphere",
        "The carbon cycle circulates carbon through atmosphere oceans organisms and geological deposits",
        "Stars fuse hydrogen into helium in their cores releasing enormous amounts of energy",
        "Black holes form when massive stars collapse and spacetime curvature prevents light escape",
        "The Big Bang theory describes the universe expanding from an extremely hot dense state",
        "Continental drift explains how continents have moved apart over geological time",
        "The water cycle describes evaporation condensation precipitation and collection of water",
        "Greenhouse gases trap infrared radiation in the atmosphere warming Earth's surface",
    ]
    count = _write_entries(out, entries, "curated/science")
    print(f"  Curated science: {count} entries")
    return count


def main():
    print(f"Preparing corpus → {OUTPUT}")
    print()

    total = 0
    with open(OUTPUT, "w", encoding="utf-8") as out:
        total += extract_swe_bench(out)
        total += extract_mbpp(out)
        total += extract_longbench(out)
        total += write_curated_ml(out)
        total += write_curated_science(out)

    print(f"\nTotal: {total} entries → {OUTPUT}")
    print(f"File size: {OUTPUT.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
