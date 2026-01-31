import ollama
import time
import requests
from typing import Dict, Any, Optional

class OllamaClient:
    """
    A wrapper around the Ollama Python client with explicit VRAM/Model management.
    Ensures that large models are unloaded before new ones are loaded to stay within VRAM limits.
    """
    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host
        self.client = ollama.Client(host=host)
        self.current_model: Optional[str] = None

    def load_model(self, model_name: str, keep_alive: str = "10m"):
        """Loads a model into VRAM."""
        if self.current_model == model_name:
            print(f"[OllamaClient] Model {model_name} already loaded.")
            return

        print(f"[OllamaClient] Loading model: {model_name}...")
        # Pre-load the model
        try:
            # We use generate with an empty prompt and keep_alive to pull it into memory
            self.client.generate(model=model_name, prompt="", keep_alive=keep_alive)
            self.current_model = model_name
            print(f"[OllamaClient] {model_name} is now active.")
        except Exception as e:
            print(f"[OllamaClient] Error loading {model_name}: {e}")
            raise

    def unload_model(self, model_name: Optional[str] = None):
        """Unloads a model from VRAM by setting keep_alive to 0."""
        target = model_name or self.current_model
        if not target:
            return

        print(f"[OllamaClient] Unloading model: {target}...")
        try:
            self.client.generate(model=target, prompt="", keep_alive=0)
            if target == self.current_model:
                self.current_model = None
            print(f"[OllamaClient] {target} unloaded.")
        except Exception as e:
            print(f"[OllamaClient] Error unloading {target}: {e}")

    def generate(self, model: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generates a response, ensuring the model is loaded."""
        if self.current_model != model:
            # If we are changing models, unload the old one first if it's large?
            # For now, let's just unload whatever is there if it's not the target.
            if self.current_model:
                self.unload_model()
            self.load_model(model)
        
        return self.client.generate(model=model, prompt=prompt, **kwargs)

    def chat(self, model: str, messages: list, **kwargs) -> Dict[str, Any]:
        """Chat completion, ensuring the model is loaded."""
        if self.current_model != model:
            if self.current_model:
                self.unload_model()
            self.load_model(model)
            
        return self.client.chat(model=model, messages=messages, **kwargs)
