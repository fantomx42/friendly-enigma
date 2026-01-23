"""
consolidator.py - Memory Consolidation Engine for REM Sleep

Clusters similar lessons using semantic similarity and synthesizes
higher-level guidelines through LLM reasoning.
"""

import os
import sys
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# Ensure ralph_core is in path
_ralph_core_path = os.path.dirname(os.path.abspath(__file__))
if _ralph_core_path not in sys.path:
    sys.path.insert(0, _ralph_core_path)

from vector_db import vector_memory, VectorDB
from agents.common.llm import call_model

# Conservative threshold - only highly similar lessons cluster
SIMILARITY_THRESHOLD = 0.80
MIN_CLUSTER_SIZE = 3


def get_lesson_embeddings(vector_db: VectorDB) -> Tuple[List[str], np.ndarray]:
    """
    Extract lesson texts and their embeddings from global vector DB.

    Returns:
        Tuple of (lesson_texts, embedding_matrix)
    """
    lessons = []
    embeddings = []

    for i, doc in enumerate(vector_db.global_docs):
        if doc.get("metadata", {}).get("tag") == "lesson":
            lessons.append(doc["text"])
            embeddings.append(vector_db.global_vectors[i])

    if not lessons:
        return [], np.array([])

    return lessons, np.array(embeddings)


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix.

    Args:
        embeddings: NxD matrix of N embeddings with D dimensions

    Returns:
        NxN similarity matrix
    """
    if embeddings.size == 0:
        return np.array([])

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    normalized = embeddings / norms

    # Cosine similarity via dot product of normalized vectors
    similarity_matrix = np.dot(normalized, normalized.T)

    return similarity_matrix


def cluster_lessons(
    lessons: List[str],
    similarity_matrix: np.ndarray,
    threshold: float = SIMILARITY_THRESHOLD
) -> List[List[str]]:
    """
    Group similar lessons into clusters using greedy agglomerative approach.

    Uses conservative threshold (0.80) to ensure only highly related
    lessons get grouped together, producing fewer but higher-quality clusters.

    Args:
        lessons: List of lesson texts
        similarity_matrix: NxN pairwise similarity matrix
        threshold: Minimum similarity to join a cluster (default: 0.80)

    Returns:
        List of clusters, each cluster is a list of lesson texts
    """
    if len(lessons) == 0:
        return []

    n = len(lessons)
    assigned = [False] * n
    clusters = []

    for i in range(n):
        if assigned[i]:
            continue

        # Start new cluster with lesson i
        cluster_indices = [i]
        assigned[i] = True

        # Find all lessons similar to any lesson in the cluster
        for j in range(i + 1, n):
            if assigned[j]:
                continue

            # Check similarity to all lessons in current cluster
            max_sim = max(similarity_matrix[k][j] for k in cluster_indices)

            if max_sim >= threshold:
                cluster_indices.append(j)
                assigned[j] = True

        # Only keep clusters with enough lessons
        if len(cluster_indices) >= MIN_CLUSTER_SIZE:
            clusters.append([lessons[idx] for idx in cluster_indices])

    return clusters


def synthesize_guideline(cluster: List[str], model: str = "orchestrator") -> Optional[Dict]:
    """
    Use LLM to create a meta-guideline from a cluster of related lessons.

    Args:
        cluster: List of similar lesson texts
        model: Model role to use for synthesis

    Returns:
        Dict with guideline text, source info, and confidence score
    """
    lessons_text = "\n".join(f"- {lesson}" for lesson in cluster)

    prompt = f"""You are a knowledge synthesizer. Given these related lessons learned from past failures, create ONE concise meta-guideline that captures the common principle.

LESSONS:
{lessons_text}

Instructions:
1. Identify the common theme or principle across all lessons
2. Create a single, actionable guideline (1-2 sentences)
3. The guideline should be general enough to apply broadly but specific enough to be useful

Output ONLY the guideline text, nothing else. Start with an action verb.
Example: "Always validate user input before processing to prevent injection attacks."

GUIDELINE:"""

    response = call_model(model, prompt)

    if not response or "ERROR" in response:
        return None

    # Clean up response
    guideline = response.strip().strip('"').strip("'")

    # Calculate confidence based on cluster cohesion
    # (In a real system, we'd compute average pairwise similarity)
    confidence = min(0.95, 0.70 + (len(cluster) * 0.05))

    return {
        "text": guideline,
        "source_count": len(cluster),
        "source_lessons": cluster,
        "confidence": round(confidence, 2),
        "created_at": datetime.now().isoformat()
    }


def categorize_guideline(guideline_text: str, model: str = "orchestrator") -> str:
    """
    Determine the category for a guideline.

    Returns one of: File Operations, Error Handling, Regex Patterns,
    Data Validation, API Calls, Testing, Security, General
    """
    prompt = f"""Categorize this programming guideline into ONE of these categories:
- File Operations
- Error Handling
- Regex Patterns
- Data Validation
- API Calls
- Testing
- Security
- General

Guideline: "{guideline_text}"

Output ONLY the category name, nothing else.

CATEGORY:"""

    response = call_model(model, prompt)

    valid_categories = [
        "File Operations", "Error Handling", "Regex Patterns",
        "Data Validation", "API Calls", "Testing", "Security", "General"
    ]

    category = response.strip()
    if category not in valid_categories:
        category = "General"

    return category


def consolidate_lessons(
    vector_db: VectorDB = None,
    threshold: float = SIMILARITY_THRESHOLD
) -> Dict:
    """
    Main consolidation function - analyzes all lessons and generates guidelines.

    Args:
        vector_db: VectorDB instance (uses global instance if None)
        threshold: Similarity threshold for clustering

    Returns:
        Dict with consolidation results
    """
    if vector_db is None:
        vector_db = vector_memory

    start_time = datetime.now()

    # Phase 1: Extract lessons and embeddings
    lessons, embeddings = get_lesson_embeddings(vector_db)

    if len(lessons) < MIN_CLUSTER_SIZE:
        return {
            "success": False,
            "error": f"Not enough lessons to consolidate (need {MIN_CLUSTER_SIZE}, have {len(lessons)})",
            "lessons_analyzed": len(lessons),
            "clusters_found": 0,
            "guidelines_created": 0
        }

    print(f"[Consolidator] Analyzing {len(lessons)} lessons...")

    # Phase 2: Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(embeddings)

    # Phase 3: Cluster similar lessons
    clusters = cluster_lessons(lessons, similarity_matrix, threshold)
    print(f"[Consolidator] Found {len(clusters)} clusters with {MIN_CLUSTER_SIZE}+ lessons")

    if not clusters:
        return {
            "success": True,
            "lessons_analyzed": len(lessons),
            "clusters_found": 0,
            "guidelines_created": 0,
            "guidelines": [],
            "message": "No clusters met the similarity threshold"
        }

    # Phase 4: Synthesize guidelines from clusters
    guidelines = []
    for i, cluster in enumerate(clusters):
        print(f"[Consolidator] Synthesizing guideline {i+1}/{len(clusters)}...")

        guideline = synthesize_guideline(cluster)
        if guideline:
            # Add category
            guideline["category"] = categorize_guideline(guideline["text"])
            guidelines.append(guideline)

    duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

    return {
        "success": True,
        "lessons_analyzed": len(lessons),
        "clusters_found": len(clusters),
        "guidelines_created": len(guidelines),
        "guidelines": guidelines,
        "duration_ms": duration_ms
    }


def format_guidelines_markdown(guidelines: List[Dict]) -> str:
    """
    Format guidelines as markdown for global_guidelines.md
    """
    # Group by category
    by_category = {}
    for g in guidelines:
        cat = g.get("category", "General")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(g)

    md = f"""# Ralph Global Guidelines

> Auto-generated meta-rules synthesized from learned lessons.
> Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

"""

    for category in sorted(by_category.keys()):
        md += f"## {category}\n\n"

        for g in by_category[category]:
            md += f"- **{g['text']}**\n"
            md += f"  - Source: {g['source_count']} lessons\n"
            md += f"  - Confidence: {g['confidence']}\n"
            md += f"  - Created: {g['created_at']}\n\n"

    return md


if __name__ == "__main__":
    # Test consolidation
    print("=== Testing Memory Consolidation ===\n")

    result = consolidate_lessons()

    print(f"\nResult: {json.dumps({k: v for k, v in result.items() if k != 'guidelines'}, indent=2)}")

    if result.get("guidelines"):
        print("\nGenerated Guidelines:")
        for g in result["guidelines"]:
            print(f"  [{g['category']}] {g['text']}")
