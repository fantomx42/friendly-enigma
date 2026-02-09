import torch
import pytest
import numpy as np
from wheeler.core.engine import DynamicsEngine

def test_engine_initialization():
    engine = DynamicsEngine(width=128, height=128)
    assert engine.width == 128
    assert engine.height == 128

def test_dynamics_step_shape():
    engine = DynamicsEngine(width=128, height=128)
    # Random input grid
    grid = torch.randn(128, 128)
    
    # Run one step
    next_grid = engine.step(grid)
    
    assert next_grid.shape == (128, 128)
    assert isinstance(next_grid, torch.Tensor)

def test_dynamics_determinism():
    engine = DynamicsEngine()
    grid = torch.randn(128, 128)
    
    # Run simulation twice
    res1 = engine.run(grid, steps=10)
    res2 = engine.run(grid, steps=10)
    
    assert torch.allclose(res1, res2), "Dynamics must be deterministic"

def test_convergence_check():
    """Verify that the engine calculates stability/convergence."""
    engine = DynamicsEngine()
    grid = torch.zeros(128, 128) # Stable state (usually)
    
    final_grid, stability = engine.run_with_stats(grid, steps=5)
    
    assert isinstance(stability, float)
    assert 0.0 <= stability <= 1.0

def test_different_inputs_diverge():
    """Verify that dynamics don't map everything to the same output immediately."""
    engine = DynamicsEngine()
    grid1 = torch.randn(128, 128)
    grid2 = torch.randn(128, 128)
    
    res1 = engine.run(grid1, steps=5)
    res2 = engine.run(grid2, steps=5)
    
    assert not torch.allclose(res1, res2)
