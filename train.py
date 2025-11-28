import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
from datasets import load_dataset
import importlib.util
import sys
import os
import time
from tqdm import tqdm

# Configuration
MODEL_PATH_V7 = "hst.v7.1_ultimate.py"
MODEL_PATH_V3 = "hst_v3_ultra.py"
BATCH_SIZE = 4  # Adjust based on VRAM
SEQ_LEN = 512
EPOCHS = 1
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

def load_model_from_path(path, model_name):
    spec = importlib.util.spec_from_file_location("hst_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["hst_module"] = module
    spec.loader.exec_module(module)
    return getattr(module, model_name)

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.inputs = []
        
        print("Tokenizing data...")
        # Simple sliding window for demonstration
        full_text = " ".join(texts)
        tokens = tokenizer.encode(full_text)
        
        for i in range(0, len(tokens) - seq_len, seq_len):
            self.inputs.append(torch.tensor(tokens[i:i+seq_len], dtype=torch.long))
            
        print(f"Created {len(self.inputs)} samples.")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]

def train(model_path, model_class_name, save_path):
    print(f"Loading model from {model_path}...")
    ModelClass = load_model_from_path(model_path, model_class_name)
    
    # Initialize Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize Model
    # Note: Adjust parameters to match the specific model's __init__
    if "HSTv7" in model_class_name:
        model = ModelClass(
            vocab_size=tokenizer.vocab_size,
            d_model=512,
            n_heads=8,
            n_layers=6,
            max_seq_len=SEQ_LEN,
            horizon=16
        )
    else: # HSTv3
        model = ModelClass(
            vocab_size=tokenizer.vocab_size,
            d_model=512,
            n_heads=8,
            n_layers=6,
            max_seq_len=SEQ_LEN,
            horizon=16
        )
        
    model.to(DEVICE)
    model.train()
    
    # Load Data
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [x['text'] for x in dataset if len(x['text']) > 0]

    train_dataset = TextDataset(texts, tokenizer, SEQ_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_idx, inputs in enumerate(progress_bar):
            inputs = inputs.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass
            # HST models return a dict
            outputs = model(inputs, training=True) if "HSTv7" in model_class_name else model(inputs)
            
            # Calculate Loss
            # We need to compute standard causal LM loss
            logits = outputs['logits']
            
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
            if batch_idx > 50: # Limit for demonstration speed
                break
                
        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(train_loader)}")
        
    print(f"Saving model to {save_path}...")
    torch.save(model.state_dict(), save_path)
    print("Training complete.")

if __name__ == "__main__":
    # Train HSTv7
    print("=== Training HSTv7.1 Ultimate ===")
    train(MODEL_PATH_V7, "HSTv7_1Ultimate", "hst_v7_checkpoint.pth")
    
    # Train HSTv3
    print("\n=== Training HSTv3 Ultra ===")
    train(MODEL_PATH_V3, "HSTv3Ultra", "hst_v3_checkpoint.pth")
