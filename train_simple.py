#!/usr/bin/env python3
"""Simplified training script for HST models with better error handling."""
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Tokenizer
from datasets import load_dataset
import sys
import time

# Import models directly
try:
    from hst_v3_ultra import HSTv3Ultra
    from hst import HSTv7_1Ultimate
except:
    # If direct import fails, try dynamic import
    import importlib.util
    
    def load_model_class(path, class_name):
        spec = importlib.util.spec_from_file_location("model_module", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, class_name)
    
    try:
        HSTv3Ultra = load_model_class("hst_v3_ultra.py", "HSTv3Ultra")
        HSTv7_1Ultimate = load_model_class("hst.v7.1_ultimate.py", "HSTv7_1Ultimate")
    except Exception as e:
        print(f"Failed to load models: {e}")
        sys.exit(1)

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
SEQ_LEN = 256  # Reduced for faster training
D_MODEL = 256  # Reduced model size
N_HEADS = 4
N_LAYERS = 4
LEARNING_RATE = 3e-4
MAX_STEPS = 100  # Limited steps for quick training

print(f"Device: {DEVICE}")
print(f"CUDA available: {torch.cuda.is_available()}")

def train_model(model_class, model_name, checkpoint_path):
    """Train a single model."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [x['text'] for x in dataset if len(x['text']) > 50][:500]  # Limit dataset
    
    # Tokenize
    print("Tokenizing...")
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text, max_length=SEQ_LEN, truncation=True)
        if len(tokens) == SEQ_LEN:
            all_tokens.append(tokens)
        if len(all_tokens) >= 200:  # Limit samples
            break
    
    print(f"Created {len(all_tokens)} training samples")
    
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
        return False
    
    model = model.to(DEVICE)
    model.train()
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    
    # Training loop
    print("Starting training...")
    start_time = time.time()
    
    for step in range(MAX_STEPS):
        # Get batch
        batch_idx = (step * BATCH_SIZE) % len(all_tokens)
        batch_tokens = all_tokens[batch_idx:batch_idx + BATCH_SIZE]
        if len(batch_tokens) < BATCH_SIZE:
            batch_tokens = all_tokens[:BATCH_SIZE]
        
        inputs = torch.tensor(batch_tokens, dtype=torch.long).to(DEVICE)
        
        # Forward pass
        optimizer.zero_grad()
        
        try:
            outputs = model(inputs)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            
            # Compute loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs[..., 1:].contiguous()
            
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if step % 10 == 0:
                print(f"Step {step}/{MAX_STEPS} | Loss: {loss.item():.4f}")
                
        except Exception as e:
            print(f"Training step failed: {e}")
            return False
    
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f}s")
    
    # Save checkpoint
    print(f"Saving to {checkpoint_path}...")
    torch.save(model.state_dict(), checkpoint_path)
    
    return True

if __name__ == "__main__":
    # Train HSTv3
    success_v3 = train_model(HSTv3Ultra, "HSTv3Ultra", "hst_v3_checkpoint.pth")
    
    # Train HSTv7
    success_v7 = train_model(HSTv7_1Ultimate, "HSTv7_1Ultimate", "hst_v7_checkpoint.pth")
    
    if success_v3 and success_v7:
        print("\n✅ All models trained successfully!")
    else:
        print("\n❌ Some models failed to train")
        sys.exit(1)
