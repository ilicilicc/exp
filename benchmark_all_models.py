#!/usr/bin/env python3
"""Comprehensive benchmark of ALL HST models - Direct speed comparison."""
import torch
import time
from transformers import GPT2Tokenizer
import importlib.util
import sys
import os

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GEN_TOKENS = 200  # Generate 200 tokens per model for fair comparison
D_MODEL = 256  # Standardized for fair comparison
N_HEADS = 4
N_LAYERS = 4
SEQ_LEN = 256

print(f"{'='*80}")
print(f"HST MODEL SPEED COMPARISON - All Models Benchmark")
print(f"{'='*80}")
print(f"Device: {DEVICE}")
print(f"Tokens to generate: {GEN_TOKENS}")
print(f"Standardized config: d_model={D_MODEL}, n_heads={N_HEADS}, n_layers={N_LAYERS}")
print()

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Model definitions
MODELS = [
    ("hst_v3_ultra.py", "HSTv3Ultra"),
    ("hst_v4_unified.py", "HSTv4Unified"),
    ("hst_v5_2_unified.py", "HSTv5_2Unified"),
    ("hst_v6_giga.py", "HSTv6Giga"),
    ("hst.v6.1.py", "HSTv6_1"),
    ("hst.v7.0_agile.py", "HSTv7_0Agile"),
    ("hst.v7.1_ultimate.py", "HSTv7_1Ultimate"),
    ("hst.v7_1_ultimate.py", "HSTv7_1Ultimate"),  # Duplicate filename
]

def load_model_class(path, class_name):
    """Dynamically load model class from file."""
    if not os.path.exists(path):
        return None
    try:
        spec = importlib.util.spec_from_file_location("model_module", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, class_name)
    except Exception as e:
        print(f"  ❌ Failed to load {path}: {e}")
        return None

def benchmark_model(model_path, model_class_name, checkpoint_path=None):
    """Benchmark a single model."""
    print(f"\n{'─'*80}")
    print(f"Testing: {model_path} ({model_class_name})")
    print(f"{'─'*80}")
    
    # Load model class
    ModelClass = load_model_class(model_path, model_class_name)
    if ModelClass is None:
        return None
    
    # Initialize model
    try:
        model = ModelClass(
            vocab_size=tokenizer.vocab_size,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            max_seq_len=SEQ_LEN,
            horizon=8
        )
        print("  ✅ Model initialized")
    except Exception as e:
        print(f"  ❌ Initialization failed: {e}")
        return None
    
    # Load checkpoint if available
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
            print(f"  ✅ Loaded checkpoint: {checkpoint_path}")
        except:
            print(f"  ⚠️  Using random weights (checkpoint load failed)")
    else:
        print(f"  ⚠️  Using random weights (no checkpoint)")
    
    model = model.to(DEVICE)
    model.eval()
    
    # Prepare prompt
    prompt_text = "The future of artificial intelligence is"
    input_ids = torch.tensor(tokenizer.encode(prompt_text)).unsqueeze(0).to(DEVICE)
    
    # Warmup
    with torch.no_grad():
        try:
            _ = model(input_ids)
        except:
            pass
    
    # Benchmark
    generated_tokens = []
    start_time = time.time()
    
    with torch.no_grad():
        current_ids = input_ids
        
        for i in range(GEN_TOKENS):
            try:
                outputs = model(current_ids)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                
                next_token_logits = logits[0, -1, :]
                next_token = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
                
                current_ids = torch.cat([current_ids, next_token], dim=1)
                generated_tokens.append(next_token.item())
                
                if current_ids.size(1) > SEQ_LEN:
                    current_ids = current_ids[:, -SEQ_LEN:]
                    
            except Exception as e:
                print(f"  ❌ Generation failed at token {i}: {e}")
                break
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Calculate metrics
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    num_chars = len(generated_text)
    num_tokens = len(generated_tokens)
    
    tps = num_tokens / duration if duration > 0 else 0
    cps = num_chars / duration if duration > 0 else 0
    
    print(f"\n  Results:")
    print(f"    Duration:   {duration:6.2f}s")
    print(f"    Tokens:     {num_tokens}")
    print(f"    Characters: {num_chars}")
    print(f"    TPS:        {tps:6.2f} tokens/sec")
    print(f"    CPS:        {cps:6.2f} chars/sec")
    
    return {
        'model': model_path,
        'class': model_class_name,
        'tps': tps,
        'cps': cps,
        'duration': duration,
        'tokens': num_tokens,
        'chars': num_chars,
        'success': num_tokens > 0
    }

# Run benchmarks
results = []

for model_path, model_class in MODELS:
    # Skip duplicate
    if model_path == "hst.v7_1_ultimate.py":
        continue
    
    # Determine checkpoint path
    checkpoint = None
    if "v3" in model_path:
        checkpoint = "hst_v3_checkpoint.pth"
    elif "v6_giga" in model_path:
        checkpoint = "hst_v6_giga_checkpoint.pth"
    elif "v7.1" in model_path or "v7_1" in model_path:
        checkpoint = "hst_v7_checkpoint.pth"
    
    result = benchmark_model(model_path, model_class, checkpoint)
    if result:
        results.append(result)

# Summary
print(f"\n{'='*80}")
print(f"BENCHMARK SUMMARY - Speed Comparison")
print(f"{'='*80}")
print()

if results:
    # Sort by TPS (descending)
    results_sorted = sorted(results, key=lambda x: x['tps'], reverse=True)
    
    print(f"{'Rank':<6} {'Model':<30} {'TPS':>10} {'CPS':>10} {'Time':>10}")
    print(f"{'─'*6} {'─'*30} {'─'*10} {'─'*10} {'─'*10}")
    
    for i, r in enumerate(results_sorted, 1):
        model_name = r['model'].replace('.py', '')
        print(f"{i:<6} {model_name:<30} {r['tps']:>10.2f} {r['cps']:>10.2f} {r['duration']:>9.2f}s")
    
    print()
    print(f"Fastest Model: {results_sorted[0]['model']} ({results_sorted[0]['tps']:.2f} TPS)")
    print(f"Slowest Model: {results_sorted[-1]['model']} ({results_sorted[-1]['tps']:.2f} TPS)")
    print(f"Speed Range: {results_sorted[0]['tps']/results_sorted[-1]['tps']:.2f}x difference")
    
    avg_tps = sum(r['tps'] for r in results) / len(results)
    avg_cps = sum(r['cps'] for r in results) / len(results)
    print(f"\nAverage TPS: {avg_tps:.2f}")
    print(f"Average CPS: {avg_cps:.2f}")
else:
    print("No successful benchmarks")

print(f"\n{'='*80}")
