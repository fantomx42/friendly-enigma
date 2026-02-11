import torch
import os
import pytest
from wheeler.core.viz import render_frame, render_comparison

def test_render_frame(tmp_path):
    frame = torch.randn(128, 128)
    output_path = tmp_path / "frame.png"
    
    render_frame(frame, str(output_path), title="Test Frame")
    assert output_path.exists()

def test_render_comparison(tmp_path):
    f1 = torch.randn(128, 128)
    f2 = torch.randn(128, 128)
    output_path = tmp_path / "comparison.png"
    
    render_comparison(f1, f2, str(output_path), title="Test Comparison")
    assert output_path.exists()

def test_animate_trajectory(tmp_path):
    from wheeler.core.viz import animate_trajectory
    trajectory = [torch.randn(128, 128) for _ in range(5)]
    output_path = tmp_path / "evolution.gif"
    
    animate_trajectory(trajectory, str(output_path), fps=5)
    assert output_path.exists()
