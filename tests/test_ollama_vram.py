import pytest
import ollama
import time

def test_ollama_basic_connection():
    """Verify that we can talk to the local Ollama server."""
    try:
        models = ollama.list()
        assert 'models' in models
        print("Connected to Ollama.")
    except Exception as e:
        pytest.fail(f"Could not connect to Ollama: {e}")

def test_explicit_unload():
    """Verify if we can request a model unload by keeping alive for 0."""
    # This is a bit hard to verify strictly via API without VRAM monitoring
    # but we can check if the call succeeds.
    try:
        # We use a tiny model if possible, or just one that should exist
        ollama.generate(model='phi3:mini', prompt='hi', keep_alive=0)
        print("Unload command (keep_alive=0) sent successfully.")
    except Exception as e:
        pytest.fail(f"Unload failed: {e}")

if __name__ == "__main__":
    test_ollama_basic_connection()
    test_explicit_unload()
