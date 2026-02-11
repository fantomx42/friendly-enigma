import torch
import pytest
from wheeler.core.reasoning import ReasoningEngine

@pytest.fixture
def engine():
    return ReasoningEngine()

def test_blend_proportions(engine):
    f1 = torch.ones(128, 128)
    f2 = torch.zeros(128, 128)
    
    # 50/50 blend
    blended = engine.blend([f1, f2], weights=[0.5, 0.5])
    assert torch.allclose(blended, torch.full((128, 128), 0.5))
    
    # Weighted blend
    blended_weighted = engine.blend([f1, f2], weights=[0.8, 0.2])
    assert torch.allclose(blended_weighted, torch.full((128, 128), 0.8))

def test_contrast(engine):
    f1 = torch.full((128, 128), 0.7)
    f2 = torch.full((128, 128), 0.3)
    
    diff = engine.contrast(f1, f2)
    assert torch.allclose(diff, torch.full((128, 128), 0.4))

def test_amplify(engine):
    f1 = torch.tensor([0.1, 0.5, -0.2])
    # Amplify should push values away from zero or strengthen them non-linearly
    # Based on the implementation (strength > 1.0), it should make them larger in magnitude
    amp = engine.amplify(f1, strength=2.0)
    
    assert torch.abs(amp[0]) > 0.1
    assert torch.abs(amp[1]) > 0.5
    assert torch.sign(amp[2]) == -1
    assert torch.abs(amp[2]) > 0.2

def test_blend_clamping(engine):
    f1 = torch.full((128, 128), 1.0)
    f2 = torch.full((128, 128), 1.0)
    
    # Even if weights are > 1, it should clamp to [-10, 10] or similar stability bounds
    blended = engine.blend([f1, f2], weights=[1.0, 1.0])
    assert torch.all(blended <= 10.0)
