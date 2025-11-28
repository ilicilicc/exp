#!/usr/bin/env python3
"""Production-grade training and benchmarking for HSTv6 Giga."""
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Tokenizer
from datasets import load_dataset
import time
import sys

# Import HSTv6 Giga
try:
    from hst_v6_giga import HSTv6Giga
except:
    import importlib.util
    spec = importlib.util.spec_from_file_location("model", "hst_v6_giga.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    HSTv6Giga = module.HSTv6Giga

# Production-like configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
SEQ_LEN = 512  # Production sequence length
D_MODEL = 512  # Production model size
N_HEADS = 8
N_LAYERS = 6
LEARNING_RATE = 1e-4
TRAIN_STEPS = 150
GEN_TOKENS = 1000  # Generate 1000 tokens for benchmark

print(f"{'='*70}")
print(f"HSTv6 Giga - Production Training & Benchmarking")
print(f"{'='*70}")
print(f"Device: {DEVICE}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Model config: d_model={D_MODEL}, n_heads={N_HEADS}, n_layers={N_LAYERS}")
print(f"Sequence length: {SEQ_LEN}")
print()

# Initialize tokenizer
print("Loading GPT-2 tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Load WikiText-2
print("Loading WikiText-2 dataset...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
texts = [x['text'] for x in dataset if len(x['text']) > 100][:1000]

# Tokenize
print("Tokenizing data...")
all_tokens = []
for text in texts:
    tokens = tokenizer.encode(text, max_length=SEQ_LEN, truncation=True)
    if len(tokens) == SEQ_LEN:
        all_tokens.append(tokens)
    if len(all_tokens) >= 300:
        break

print(f"Created {len(all_tokens)} training samples")
print()

# Initialize model
print("Initializing HSTv6 Giga model...")
try:
    model = HSTv6Giga(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        max_seq_len=SEQ_LEN,
        horizon=16
    )
    print("✅ Model initialized successfully")
except Exception as e:
    print(f"❌ Model initialization failed: {e}")
    sys.exit(1)

model = model.to(DEVICE)
print(f"Model moved to {DEVICE}")
print()

# Training
print(f"{'='*70}")
print("TRAINING")
print(f"{'='*70}")
model.train()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

start_time = time.time()
losses = []

for step in range(TRAIN_STEPS):
    batch_idx = (step * BATCH_SIZE) % len(all_tokens)
    batch_tokens = all_tokens[batch_idx:batch_idx + BATCH_SIZE]
    if len(batch_tokens) < BATCH_SIZE:
        batch_tokens = all_tokens[:BATCH_SIZE]
    
    inputs = torch.tensor(batch_tokens, dtype=torch.long).to(DEVICE)
    
    optimizer.zero_grad()
    
    try:
        outputs = model(inputs)
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs[..., 1:].contiguous()
        
        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 20 == 0:
            print(f"Step {step:3d}/{TRAIN_STEPS} | Loss: {loss.item():.4f}")
            
    except Exception as e:
        print(f"❌ Training failed at step {step}: {e}")
        sys.exit(1)

train_time = time.time() - start_time
print(f"\n✅ Training completed in {train_time:.2f}s")
print(f"Final loss: {losses[-1]:.4f} (started at {losses[0]:.4f})")

# Save checkpoint
checkpoint_path = "hst_v6_giga_checkpoint.pth"
print(f"Saving checkpoint to {checkpoint_path}...")
torch.save(model.state_dict(), checkpoint_path)
print("✅ Checkpoint saved")
print()

# Benchmarking
print(f"{'='*70}")
print("BENCHMARKING - Production TPS/CPS Test")
print(f"{'='*70}")
model.eval()

prompt_text = "The rapid advancement of artificial intelligence and machine learning technologies has transformed"
input_ids = torch.tensor(tokenizer.encode(prompt_text)).unsqueeze(0).to(DEVICE)

print(f"Prompt: {prompt_text}")
print(f"Generating {GEN_TOKENS} tokens...")
print()

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
                
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                current_tps = (i + 1) / elapsed
                print(f"  {i+1:4d} tokens | {elapsed:6.2f}s | TPS: {current_tps:7.2f}")
                
        except Exception as e:
            print(f"❌ Generation failed at token {i}: {e}")
            break

end_time = time.time()
duration = end_time - start_time

# Results
generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
full_text = prompt_text + generated_text

num_chars = len(generated_text)
num_tokens = len(generated_tokens)
tps = num_tokens / duration if duration > 0 else 0
cps = num_chars / duration if duration > 0 else 0

print()
print(f"{'='*70}")
print("RESULTS")
print(f"{'='*70}")
print(f"Generated text preview:")
print(f"{full_text[:300]}...")
print()
print(f"Performance Metrics:")
print(f"  Duration:              {duration:.2f}s")
print(f"  Tokens generated:      {num_tokens}")
print(f"  Characters generated:  {num_chars}")
print(f"  TPS (Tokens/sec):      {tps:.2f}")
print(f"  CPS (Chars/sec):       {cps:.2f}")
print()

if cps > 5000:
    print(f"  ✅ TARGET ACHIEVED: CPS > 5000!")
else:
    print(f"  ❌ Target not met (need >5000 CPS, got {cps:.2f})")
    print(f"  Note: GPU would provide 10-100x speedup")

print(f"{'='*70}")
