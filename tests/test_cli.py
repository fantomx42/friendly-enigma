import pytest
from click.testing import CliRunner
import os
from wheeler.cli import main

@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture
def storage_dir(tmp_path):
    d = tmp_path / "cli_storage"
    d.mkdir()
    return str(d)

def test_cli_store(runner, storage_dir):
    env = {"WHEELER_STORAGE": storage_dir}
    result = runner.invoke(main, ["store", "Hello CLI"], env=env)
    assert result.exit_code == 0
    assert "Stored memory" in result.output
    assert "UUID:" in result.output

def test_cli_recall(runner, storage_dir):
    env = {"WHEELER_STORAGE": storage_dir}
    # Store first
    runner.invoke(main, ["store", "The cat is back"], env=env)
    
    # Recall
    result = runner.invoke(main, ["recall", "cat"], env=env)
    assert result.exit_code == 0
    assert "Results:" in result.output
    assert "The cat is back" in result.output
    assert "Similarity:" in result.output

def test_cli_reason(runner, storage_dir):
    env = {"WHEELER_STORAGE": storage_dir}
    runner.invoke(main, ["store", "Fire"], env=env)
    runner.invoke(main, ["store", "Water"], env=env)
    
    result = runner.invoke(main, ["reason", "Fire", "Water"], env=env)
    assert result.exit_code == 0
    assert "Inferred Associations:" in result.output

def test_cli_dream(runner, storage_dir):
    env = {"WHEELER_STORAGE": storage_dir}
    runner.invoke(main, ["store", "A"], env=env)
    runner.invoke(main, ["store", "B"], env=env)
    
    result = runner.invoke(main, ["dream", "--ticks", "2"], env=env)
    assert result.exit_code == 0
    assert "Autonomic cycle complete." in result.output
