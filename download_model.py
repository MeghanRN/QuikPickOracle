"""Download the local Llama model if it is missing.

This script fetches the quantized GGUF model from Hugging Face and stores it
under ``model/`` so ``src/app.py`` can load it with ``llama_cpp``. Run it once
before launching the Streamlit app.
"""
from pathlib import Path

from huggingface_hub import hf_hub_download

MODEL_REPO = "TheBloke/Llama-3-13B-Instruct-GGUF"
MODEL_FILE = "Llama-3-13B-Instruct-Q4_K_M.gguf"
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / MODEL_FILE

def ensure_model() -> Path:
    """Download the model file if missing and return its path."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            local_dir=str(MODEL_DIR),
            local_dir_use_symlinks=False,
        )
    return MODEL_PATH

if __name__ == "__main__":
    path = ensure_model()
    print(f"Model available at {path}")
