import torch
import time
from transformers import GPT2Tokenizer
import importlib.util
import sys
import os

# Configuration
MODEL_PATH_V7 = "hst.v7.1_ultimate.py"
MODEL_PATH_V3 = "hst_v3_ultra.py"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 512

def load_model_from_path(path, model_name):
    spec = importlib.util.spec_from_file_location("hst_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["hst_module"] = module
    spec.loader.exec_module(module)
    return getattr(module, model_name)

def benchmark(model_path, model_class_name, checkpoint_path):
    print(f"Benchmarking {model_class_name}...")
    ModelClass = load_model_from_path(model_path, model_class_name)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Initialize Model
    if "HSTv7" in model_class_name:
        model = ModelClass(
            vocab_size=tokenizer.vocab_size,
            d_model=512,
            n_heads=8,
            n_layers=6,
            max_seq_len=SEQ_LEN,
            horizon=16
        )
    else:
        model = ModelClass(
            vocab_size=tokenizer.vocab_size,
            d_model=512,
            n_heads=8,
            n_layers=6,
            max_seq_len=SEQ_LEN,
            horizon=16
        )
        
    # Load Checkpoint
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        print("Checkpoint loaded.")
    except FileNotFoundError:
        print("Checkpoint not found, using random weights.")
        
    model.to(DEVICE)
    model.eval()
    
    # Benchmark Generation
    prompt_text = "The future of AI is"
    input_ids = torch.tensor(tokenizer.encode(prompt_text)).unsqueeze(0).to(DEVICE)
    
    print("Warming up...")
    with torch.no_grad():
        if "HSTv7" in model_class_name:
            _ = model.generate_speculative(input_ids, max_new_tokens=10)
        else:
            _ = model.generate_ultra_fast(input_ids, max_new_tokens=10)
            
    print("Starting benchmark...")
    start_time = time.time()
    num_tokens = 200
    
    with torch.no_grad():
        if "HSTv7" in model_class_name:
            output_ids = model.generate_speculative(input_ids, max_new_tokens=num_tokens)
        else:
            output_ids, stats = model.generate_ultra_fast(input_ids, max_new_tokens=num_tokens)
            
    end_time = time.time()
    duration = end_time - start_time
    
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    new_text = generated_text[len(prompt_text):]
    num_chars = len(new_text)
    
    cps = num_chars / duration
    tps = num_tokens / duration
    
    print(f"Generated Text: {generated_text[:100]}...")
    print(f"Time: {duration:.4f}s")
    print(f"Tokens: {num_tokens}")
    print(f"Characters: {num_chars}")
    print(f"TPS: {tps:.2f}")
    print(f"CPS: {cps:.2f}")
    
    if cps > 5000:
        print("✅ Target > 5000 CPS ACHIEVED!")
    else:
        print(f"❌ Target > 5000 CPS NOT ACHIEVED (Current: {cps:.2f})")

if __name__ == "__main__":
    benchmark(MODEL_PATH_V7, "HSTv7_1Ultimate", "hst_v7_checkpoint.pth")
    benchmark(MODEL_PATH_V3, "HSTv3Ultra", "hst_v3_checkpoint.pth")
