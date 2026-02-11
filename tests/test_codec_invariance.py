import torch
import pytest
from wheeler.core.codec import TextCodec

def test_position_invariance():
    codec = TextCodec()
    
    # "A B" vs "B A"
    # In a Bag-of-Words model, these should be identical
    enc1 = codec.encode("A B")
    enc2 = codec.encode("B A")
    
    # Calculate cosine similarity
    f1 = enc1.view(-1)
    f2 = enc2.view(-1)
    sim = (f1 @ f2) / (f1.norm() * f2.norm())
    
    print(f"Similarity 'A B' vs 'B A': {sim.item()}")
    assert sim.item() > 0.99, "Encoding should be position invariant (Bag of Words)"

def test_containment():
    codec = TextCodec()
    
    # "memory" vs "permanent memory"
    # "memory" should be a component of "permanent memory"
    enc_part = codec.encode("memory")
    enc_full = codec.encode("permanent memory")
    
    f1 = enc_part.view(-1)
    f2 = enc_full.view(-1)
    sim = (f1 @ f2) / (f1.norm() * f2.norm())
    
    print(f"Similarity 'memory' vs 'permanent memory': {sim.item()}")
    # It won't be 1.0, but should be significant.
    # If "permanent memory" has 2 words, and we match 1, similarity ~ 1/sqrt(2) = 0.707
    assert sim.item() > 0.5, "Part should have high similarity to whole"
