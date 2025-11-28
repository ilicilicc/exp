#!/usr/bin/env python3
"""Simplified benchmarking script for HST models."""
import torch
import time
from transformers import GPT2Tokenizer
import sys

# Import models
try:
    from hst_v3_ultra import HSTv3Ultra
    from hst import HSTv7_1Ultimate
except:
    import importlib.util
    
    def load_model_class(path, class_name):
        spec = importlib.util.spec_from_file_location("model_module", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, class_name)
    
    HSTv3Ultra = load_model_class("hst_v3_ultra.py", "HSTv3Ultra")
    HSTv7_1Ultimate = load_model_class("hst.v7.1_ultimate.py", "HSTv7_1Ultimate")

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 256
D_MODEL = 256
N_HEADS = 4
N_LAYERS = 4
NUM_TOKENS_TO_GENERATE = 500  # Generate 500 tokens

print(f"Device: {DEVICE}")
print(f"CUDA available: {torch.cuda.is_available()}")

def benchmark_model(model_class, model_name, checkpoint_path):
    """Benchmark a single model for CPS."""
    print(f"\n{'='*60}")
    print(f"Benchmarking {model_name}")
    print(f"{'='*60}")
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Initialize model
    print("Initializing model...")
    try:
        model = model_class(
            vocab_size=tokenizer.vocab_size,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            max_seq_len=SEQ_LEN,
            horizon=8
        )
    except Exception as e:
        print(f"Model initialization failed: {e}")
        return None
    
    # Load checkpoint
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        print("Checkpoint loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load checkpoint ({e}), using random weights")
    
    model = model.to(DEVICE)
    model.eval()
    
    # Prepare prompt
    prompt_text = "The future of artificial intelligence and machine learning is"
    input_ids = torch.tensor(tokenizer.encode(prompt_text)).unsqueeze(0).to(DEVICE)
    
    print(f"Prompt: {prompt_text}")
    print(f"Generating {NUM_TOKENS_TO_GENERATE} tokens...")
    
    # Warmup
    with torch.no_grad():
        try:
            _ = model(input_ids)
        except:
            pass
    
    # Benchmark generation
    generated_tokens = []
    start_time = time.time()
    
    with torch.no_grad():
        current_ids = input_ids
        
        for i in range(NUM_TOKENS_TO_GENERATE):
            try:
                # Forward pass
                outputs = model(current_ids)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                
                # Sample next token
                next_token_logits = logits[0, -1, :]
                next_token = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
                
                # Append
                current_ids = torch.cat([current_ids, next_token], dim=1)
                generated_tokens.append(next_token.item())
                
                # Keep sequence length manageable
                if current_ids.size(1) > SEQ_LEN:
                    current_ids = current_ids[:, -SEQ_LEN:]
                    
            except Exception as e:
                print(f"Generation failed at token {i}: {e}")
                break
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Decode generated text
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    full_text = prompt_text + generated_text
    
    # Calculate metrics
    num_chars = len(generated_text)
    num_tokens = len(generated_tokens)
    
    tps = num_tokens / duration if duration > 0 else 0
    cps = num_chars / duration if duration > 0 else 0
    
    # Display results
    print(f"\nGenerated text preview:")
    print(f"{full_text[:200]}...")
    print(f"\nMetrics:")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Tokens generated: {num_tokens}")
    print(f"  Characters generated: {num_chars}")
    print(f"  TPS (Tokens/sec): {tps:.2f}")
    print(f"  CPS (Chars/sec): {cps:.2f}")
    
    if cps > 5000:
        print(f"  ✅ TARGET ACHIEVED: CPS > 5000!")
    else:
        print(f"  ❌ Target not met (need >5000 CPS, got {cps:.2f})")
    
    return {
        'model': model_name,
        'tps': tps,
        'cps': cps,
        'duration': duration,
        'tokens': num_tokens,
        'chars': num_chars
    }

if __name__ == "__main__":
    results = []
    
    # Benchmark HSTv3
    result_v3 = benchmark_model(HSTv3Ultra, "HSTv3Ultra", "hst_v3_checkpoint.pth")
    if result_v3:
        results.append(result_v3)
    
    # Benchmark HSTv7
    result_v7 = benchmark_model(HSTv7_1Ultimate, "HSTv7_1Ultimate", "hst_v7_checkpoint.pth")
    if result_v7:
        results.append(result_v7)
    
    # Summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"{r['model']:20s} | TPS: {r['tps']:8.2f} | CPS: {r['cps']:8.2f}")
    
    if results:
        avg_cps = sum(r['cps'] for r in results) / len(results)
        print(f"\nAverage CPS: {avg_cps:.2f}")
