import importlib.util
import sys
import os
import torch

MODEL_PATH = "hst.v7.1_ultimate.py"

def load_model():
    print(f"Loading {MODEL_PATH}...")
    try:
        spec = importlib.util.spec_from_file_location("hst_module", MODEL_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules["hst_module"] = module
        spec.loader.exec_module(module)
        print("Module loaded successfully.")
        return module.HSTv7_1Ultimate
    except Exception as e:
        print(f"Failed to load module: {e}")
        return None

if __name__ == "__main__":
    print("Starting debug load...")
    ModelClass = load_model()
    if ModelClass:
        print("Initializing model...")
        try:
            model = ModelClass(
                vocab_size=50257,
                d_model=512,
                n_heads=8,
                n_layers=6,
                max_seq_len=512,
                horizon=16
            )
            print("Model initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize model: {e}")
    print("Done.")
