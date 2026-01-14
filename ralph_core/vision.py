import base64
import os
import requests
from typing import Optional

OLLAMA_API = "http://localhost:11434/api/generate"
VISION_MODEL = "llava"

class VisionModule:
    def __init__(self):
        pass

    def _encode_image(self, image_path: str) -> Optional[str]:
        """Encodes an image file to base64 string."""
        if not os.path.exists(image_path):
            print(f"[Vision] Error: Image not found at {image_path}")
            return None
        
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"[Vision] Encoding error: {e}")
            return None

    def analyze_image(self, image_path: str, prompt: str = "Describe this image in detail.") -> str:
        """
        Sends an image to the Vision Model (LLaVA) for analysis.
        """
        b64_image = self._encode_image(image_path)
        if not b64_image:
            return "Error: Could not process image."

        print(f"[Vision] Analyzing {image_path} with {VISION_MODEL}...")
        
        try:
            response = requests.post(
                OLLAMA_API,
                json={
                    "model": VISION_MODEL,
                    "prompt": prompt,
                    "images": [b64_image],
                    "stream": False
                },
                timeout=120
            )
            response.raise_for_status()
            result = response.json().get("response", "").strip()
            return f"[VISION ANALYSIS]: {result}"
            
        except Exception as e:
            return f"[Vision] API Error: {str(e)}"

# Global instance
vision = VisionModule()
