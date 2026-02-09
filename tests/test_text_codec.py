import numpy as np
import pytest
import torch
from wheeler.core.codec import TextCodec

def test_text_codec_initialization():
    codec = TextCodec(width=128, height=128)
    assert codec.width == 128
    assert codec.height == 128

def test_encoding_shape_and_type():
    codec = TextCodec()
    text = "Hello, Wheeler!"
    encoded = codec.encode(text)
    
    # Expecting a torch tensor
    assert isinstance(encoded, torch.Tensor)
    assert encoded.shape == (128, 128)
    assert encoded.dtype == torch.float32

def test_determinism():
    codec = TextCodec()
    text = "Determinism Check"
    
    run1 = codec.encode(text)
    run2 = codec.encode(text)
    
    assert torch.allclose(run1, run2), "Encoding must be deterministic"

def test_sensitivity():
    codec = TextCodec()
    text1 = "Hello"
    text2 = "hello" # Case difference
    
    enc1 = codec.encode(text1)
    enc2 = codec.encode(text2)
    
    assert not torch.allclose(enc1, enc2), "Different text must produce different encodings"

def test_empty_string():
    codec = TextCodec()
    enc = codec.encode("")
    assert torch.all(enc == 0), "Empty string should produce zero grid (or specific baseline)"
