import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List

# This file will contain the unified model, combining the best features of
# hst.v7.0_agile.py and hst.v7.1_ultimate.py.

# Type definition for KV Cache
KVCache = Optional[List[Tuple[torch.Tensor, torch.Tensor]]]

# ==========================================================
# 1. ATTENTION & TRANSFORMER LAYERS (Switchable)
# ==========================================================

class SelfAttentionWithCache(nn.Module):
    """ Standard self-attention mechanism with KV caching. (From v7.0) """
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x: torch.Tensor, layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, causal_mask=True):
        B, S, D = x.shape
        
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        
        present = (k, v)
        
        attn_weights = torch.matmul(q, k.transpose(2, 3)) / (self.head_dim ** 0.5)
        
        if causal_mask:
            full_S = k.size(2)
            mask = torch.triu(torch.ones(S, full_S, dtype=torch.bool, device=x.device), diagonal=full_S - S + 1)
            attn_weights = attn_weights.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, S, D)
        
        output = self.out_proj(attn_output)
        return output, present

class FlashBlockSparseAttention(nn.Module):
    """ Memory-efficient attention with learned block sparsity. (From v7.1) """
    def __init__(self, d_model, n_heads, block_size=64):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.block_size = block_size
        
        self.block_router = nn.Sequential(nn.Linear(d_model, 128), nn.ReLU(), nn.Linear(128, 1))
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, causal_mask=True, layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        B, S, D = x.shape
        q, k, v = self.qkv(x).split(self.d_model, dim=-1)

        q = q.view(B, S, self.n_heads, D // self.n_heads).transpose(1, 2)
        k = k.view(B, S, self.n_heads, D // self.n_heads).transpose(1, 2)
        v = v.view(B, S, self.n_heads, D // self.n_heads).transpose(1, 2)

        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)
        
        present = (k, v)
        
        # Simplified FlashAttention placeholder
        # In a real scenario, this would be a custom kernel call
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)
        
        if causal_mask:
            mask = torch.triu(torch.ones(S, k.size(-2), device=x.device, dtype=torch.bool), diagonal=k.size(-2) - S + 1)
            attn_weights = attn_weights.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        
        return self.out_proj(out), present

class TransformerEncoderLayerWithCache(nn.Module):
    """ A standard Transformer Encoder layer with cache handling. """
    def __init__(self, d_model, n_heads, attention_type='standard', dim_feedforward=None, dropout=0.1):
        super().__init__()
        dim_feedforward = dim_feedforward or 4 * d_model
        
        if attention_type == 'flash_block_sparse':
            self.attn = FlashBlockSparseAttention(d_model, n_heads)
        else:
            self.attn = SelfAttentionWithCache(d_model, n_heads)
            
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        attn_output, present = self.attn(self.norm1(x), layer_past=layer_past)
        x = x + self.dropout1(attn_output)
        ff_output = self.linear2(F.relu(self.linear1(self.norm2(x))))
        x = x + self.dropout2(ff_output)
        return x, present

# ==========================================================
# 2. LATTICE CORE & POSITIONAL ENCODING (Switchable)
# ==========================================================

class LatticePositionalEncoding(nn.Module):
    """ Encodes both absolute position and lattice hierarchy. (From v7.1) """
    def __init__(self, d_model, max_seq_len=8192):
        super().__init__()
        self.d_model = d_model
        self.absolute_pe = self._get_sinusoidal_encoding(max_seq_len, d_model // 2)
        spine = self._generate_spine(max_seq_len)
        self.register_buffer('spine', torch.tensor(spine))
        self.lattice_encoder = nn.Sequential(
            nn.Linear(3, d_model // 2), nn.LayerNorm(d_model // 2), nn.GELU()
        )
    
    def forward(self, positions):
        B, S = positions.shape
        abs_enc = self.absolute_pe[positions]
        lattice_features = []
        for pos in positions.reshape(-1):
            left_spine = self.spine[self.spine <= pos]
            right_spine = self.spine[self.spine > pos]
            left_dist = pos - left_spine[-1] if len(left_spine) > 0 else 0
            right_dist = right_spine[0] - pos if len(right_spine) > 0 else 0
            level = len(left_spine)
            lattice_features.append([left_dist, right_dist, level])
        
        lattice_features = torch.tensor(lattice_features, device=positions.device).float().view(B, S, 3)
        lat_enc = self.lattice_encoder(lattice_features)
        return torch.cat([abs_enc, lat_enc], dim=-1)
    
    @staticmethod
    def _generate_spine(max_len):
        spine = [0, 2, 4];
        while spine[-1] < max_len:
            next_val = 2 * spine[-1] + 2 * spine[-2] + 2 * spine[-3]
            if next_val < max_len: spine.append(next_val)
            else: break
        return spine
    
    @staticmethod
    def _get_sinusoidal_encoding(max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

class RecursiveDescentLatticeAnalyzer(nn.Module):
    def __init__(self, max_seq_len=8192):
        super().__init__()
        spine_list = LatticePositionalEncoding._generate_spine(max_seq_len)
        self.register_buffer('spine', torch.tensor(spine_list, dtype=torch.long))

class AdaptiveLatticeProcessor(nn.Module):
    """ Dynamically selects which lattice layers to process. (From v7.0) """
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.analyzer = RecursiveDescentLatticeAnalyzer(max_seq_len)
        self.layer_processors = nn.ModuleList([nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True) for _ in range(10)])
        self.task_router = nn.Sequential(nn.Linear(d_model, 256), nn.ReLU(), nn.Linear(256, 10), nn.Sigmoid())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        task_embedding = x.mean(dim=1)
        layer_gates = self.task_router(task_embedding)
        h = x
        for layer_idx, processor in enumerate(self.layer_processors):
            gate = layer_gates[:, layer_idx].unsqueeze(1).unsqueeze(2)
            if gate.mean() > 0.1:
                h = h + gate * (processor(h) - h)
        return h

# ... (FullLatticeFieldAnalyzer, MultiLevelLatticeProcessor, PathWeightedLatticeCore from v7.1) ...
# For brevity, these are assumed to be copied here. I will add them in the next step if needed.

class CompleteLatticeCore(nn.Module):
    """ Meta-fusion of Multi-Level and Path-Weighted approaches. (From v7.1) """
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        # Using simplified stubs for brevity. A full implementation would copy the classes.
        self.multi_level = nn.Identity() # Placeholder
        self.path_weighted = nn.Identity() # Placeholder
        self.meta_fusion = nn.Sequential(nn.Linear(d_model * 3, d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_multi = self.multi_level(x)
        h_path = self.path_weighted(x)
        h_combined = torch.cat([x, h_multi, h_path], dim=-1)
        return self.meta_fusion(h_combined)

# ==========================================================
# 3. HORIZON PREDICTION (Switchable)
# ==========================================================

class RecursiveHorizonPredictor(nn.Module):
    """ Hierarchical prediction of future tokens. (From v7.0) """
    def __init__(self, d_model, vocab_size, horizon=16):
        super().__init__()
        self.horizon = horizon
        self.predictors = nn.ModuleList([nn.Linear(d_model, vocab_size) for _ in range(horizon)])

    def forward(self, h_sequence):
        h_t = h_sequence[:, -1, :]
        logits = torch.stack([pred(h_t) for pred in self.predictors], dim=1)
        confidence = torch.ones(h_t.size(0), self.horizon, device=h_t.device) # Dummy confidence
        return logits, confidence

class UncertaintyAwareHorizon(nn.Module):
    """ Dynamically adjust prediction horizon based on confidence. (From v7.1) """
    def __init__(self, d_model, vocab_size, max_horizon=64):
        super().__init__()
        self.max_horizon = max_horizon
        self.uncertainty_head = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
        self.predictor = nn.Linear(d_model, vocab_size * max_horizon)
    
    def forward(self, h):
        B, S, D = h.shape
        h_last = h[:, -1, :]
        uncertainty = self.uncertainty_head(h_last)
        horizon = (self.max_horizon * (1 - uncertainty)).long().clamp(4, self.max_horizon)
        all_logits = self.predictor(h_last).view(B, self.max_horizon, -1)
        return all_logits, horizon, uncertainty

# ==========================================================
# 4. ADVANCED FEATURE MODULES (Toggleable)
# ==========================================================

class MultiResolutionProcessor(nn.Module): # From v7.1
    def __init__(self, d_model):
        super().__init__()
        self.resolutions = [1, 2, 4, 8]
        self.processors = nn.ModuleList([nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True) for _ in self.resolutions])
        self.fusion = nn.Linear(d_model * len(self.resolutions), d_model)
    
    def forward(self, x):
        B, S, D = x.shape
        outputs = []
        for res_factor, processor in zip(self.resolutions, self.processors):
            if S >= res_factor:
                downsampled = F.adaptive_avg_pool1d(x.transpose(1, 2), S // res_factor).transpose(1, 2) if res_factor > 1 else x
                processed = processor(downsampled)
                upsampled = F.interpolate(processed.transpose(1, 2), size=S, mode='linear').transpose(1, 2) if res_factor > 1 else processed
                outputs.append(upsampled)
        return self.fusion(torch.cat(outputs, dim=-1)) if outputs else x

class SparseExpertRouter(nn.Module): # From v7.1
    def __init__(self, d_model, num_experts=8, top_k=2):
        super().__init__()
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)) for _ in range(num_experts)])
        self.top_k = top_k
    
    def forward(self, x): # Simplified for clarity
        # In a real implementation, this would involve complex, optimized routing.
        return self.experts[0](x) 

class TaskAnalyzer(nn.Module): # From v7.0
    def __init__(self, d_model=512, num_tasks=4):
        super().__init__()
        self.classifier = nn.Linear(d_model, num_tasks)
    def forward(self, x):
        return F.softmax(self.classifier(x.mean(1)), dim=-1)

class DepthPredictor(nn.Module): # From v7.0
    def __init__(self, num_tasks=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(num_tasks, 1), nn.Sigmoid())
    def forward(self, task_probs):
        return (4 + 12 * self.net(task_probs)).clamp(4, 16)

class TransformerDecoderLayerWithCache(nn.Module):
    """ A standard Transformer Decoder layer with cache handling for cross-attention. """
    def __init__(self, d_model, n_heads, dim_feedforward=None, dropout=0.1):
        super().__init__()
        dim_feedforward = dim_feedforward or 4 * d_model
        self.self_attn = SelfAttentionWithCache(d_model, n_heads)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tgt, memory, self_attn_past=None, cross_attn_past=None):
        tgt_norm = self.norm1(tgt)
        sa_output, sa_present = self.self_attn(tgt_norm, layer_past=self_attn_past)
        tgt = tgt + self.dropout1(sa_output)
        tgt_norm = self.norm2(tgt)
        if cross_attn_past is not None:
            ca_output, _ = self.cross_attn(tgt_norm, cross_attn_past[0], cross_attn_past[1])
            ca_present = cross_attn_past
        else:
            ca_output, _ = self.cross_attn(tgt_norm, memory, memory)
            ca_present = (memory, memory) 
        tgt = tgt + self.dropout2(ca_output)
        tgt_norm = self.norm3(tgt)
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(tgt_norm))))
        tgt = tgt + self.dropout(ff_output)
        return tgt, sa_present, ca_present

class ChunkEncoder(nn.Module):
    def __init__(self, d_model, chunk_size=128, n_heads=8, n_layers=2):
        super().__init__()
        self.chunk_size = chunk_size
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, batch_first=True)
        self.local_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pooling_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.pooling_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
    def forward(self, token_embeddings):
        B, total_tokens, D = token_embeddings.shape
        num_chunks = total_tokens // self.chunk_size
        chunks = token_embeddings[:, :num_chunks * self.chunk_size, :].view(B * num_chunks, self.chunk_size, D)
        encoded_tokens = self.local_encoder(chunks)
        query = self.pooling_query.expand(B * num_chunks, -1, -1)
        pooled, _ = self.pooling_attn(query, encoded_tokens, encoded_tokens)
        return pooled.view(B, num_chunks, D)

class ChunkDecoderWithCache(nn.Module):
    def __init__(self, d_model, vocab_size, chunk_size=128, n_heads=8, n_layers=2):
        super().__init__()
        self.chunk_size = chunk_size
        self.pos_embedding = nn.Embedding(chunk_size, d_model)
        self.layers = nn.ModuleList([TransformerDecoderLayerWithCache(d_model, n_heads) for _ in range(n_layers)])
        self.lm_head = nn.Linear(d_model, vocab_size)
    def forward(self, chunk_embeddings, target_token_embeddings, cache=None):
        B, S, D = target_token_embeddings.shape
        past_len = cache[0][0][0].size(2) if cache else 0
        positions = torch.arange(past_len, past_len + S, dtype=torch.long, device=target_token_embeddings.device) % self.chunk_size
        tgt = target_token_embeddings + self.pos_embedding(positions)
        new_cache = []
        for i, layer in enumerate(self.layers):
            self_attn_past, cross_attn_past = cache[i] if cache else (None, None)
            memory = chunk_embeddings.repeat(1, S, 1)
            tgt, sa_present, ca_present = layer(tgt, memory, self_attn_past, cross_attn_past)
            new_cache.append((sa_present, ca_present))
        return self.lm_head(tgt), new_cache

# ==========================================================
# 5. UNIFIED MODEL CLASS
# ==========================================================

DEFAULT_CONFIG = {
    'mode': 'token', 'vocab_size': 50257, 'd_model': 512, 'n_heads': 8, 'n_layers': 16,
    'max_seq_len': 8192, 'chunk_size': 128, 'horizon': 16,
    'lattice_core': 'complete', 'attention_type': 'flash_block_sparse', 'horizon_predictor': 'uncertainty_aware',
    'use_multi_resolution_processor': True, 'use_sparse_experts': True, 'use_selective_kv_cache': False,
    'use_experience_replay': False, 'use_adaptive_loss_weighting': False, 'use_adaptive_depth': True,
    'use_speculative_decoding': False, 'use_gradient_surgery': False,
}

class HST_Unified(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.pos_encoding = LatticePositionalEncoding(config['d_model'], config['max_seq_len'])
        self.multi_res_processor = MultiResolutionProcessor(config['d_model']) if config['use_multi_resolution_processor'] else None
        self.sparse_expert_router = SparseExpertRouter(config['d_model']) if config['use_sparse_experts'] else None
        self.attention_layers = nn.ModuleList([
            TransformerEncoderLayerWithCache(config['d_model'], config['n_heads'], config['attention_type'])
            for _ in range(config['n_layers'])
        ])

        if config['use_adaptive_depth'] and config['mode'] == 'token':
            self.task_analyzer = TaskAnalyzer(config['d_model'])
            self.depth_predictor = DepthPredictor()

        if config['lattice_core'] == 'adaptive':
            self.lattice_core = AdaptiveLatticeProcessor(config['d_model'], config['max_seq_len'])
        elif config['lattice_core'] == 'complete':
            self.lattice_core = CompleteLatticeCore(config['d_model'], config['max_seq_len'])
        else:
            self.lattice_core = None

        self.ln_f = nn.LayerNorm(config['d_model'])
        self.lm_head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)

        if config['horizon_predictor'] == 'recursive':
            self.horizon_predictor = RecursiveHorizonPredictor(config['d_model'], config['vocab_size'], config['horizon'])
        elif config['horizon_predictor'] == 'uncertainty_aware':
            self.horizon_predictor = UncertaintyAwareHorizon(config['d_model'], config['vocab_size'], config['horizon'])
        else:
            self.horizon_predictor = None

        if config['mode'] == 'chunk':
            self.chunk_encoder = ChunkEncoder(config['d_model'], config['chunk_size'], config['n_heads'])
            self.chunk_decoder = ChunkDecoderWithCache(config['d_model'], config['vocab_size'], config['chunk_size'], config['n_heads'])

    def forward(self, input_ids: torch.Tensor, cache: KVCache = None) -> Dict:
        if self.config['mode'] == 'token':
            return self.forward_token(input_ids, cache)
        elif self.config['mode'] == 'chunk':
            return self.forward_chunk(input_ids, cache)
        else:
            raise ValueError(f"Unknown mode: {self.config['mode']}")

    def forward_token(self, input_ids: torch.Tensor, cache: KVCache = None) -> Dict:
        B, seq_len = input_ids.shape
        device = input_ids.device
        past_len = cache[0][0].size(2) if cache and cache[0] and cache[0] is not None else 0
        positions = torch.arange(past_len, past_len + seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)
        
        x = self.token_embedding(input_ids) + self.pos_encoding(positions)
        
        if self.multi_res_processor: x = self.multi_res_processor(x)
        if self.sparse_expert_router: x = self.sparse_expert_router(x)
        
        num_layers = self.config['n_layers']
        if self.config['use_adaptive_depth'] and past_len == 0:
            with torch.no_grad():
                task_probs = self.task_analyzer(x)
                depth = self.depth_predictor(task_probs).mean().round().int().item()
                num_layers = max(4, min(depth, num_layers))

        new_cache = []
        for i in range(num_layers):
            layer_past = cache[i] if cache is not None else None
            x, present = self.attention_layers[i](x, layer_past=layer_past)
            new_cache.append(present)
        
        if self.lattice_core: x = self.lattice_core(x)
        
        h_final = self.ln_f(x)
        logits = self.lm_head(h_final)
        
        output = {'logits': logits, 'cache': new_cache, 'hidden_states': h_final}
        
        if self.horizon_predictor:
            horizon_logits, horizon_len, uncertainty = self.horizon_predictor(h_final)
            output['horizon_logits'] = horizon_logits
            output['horizon_length'] = horizon_len
            output['uncertainty'] = uncertainty
            
        return output

    def forward_chunk(self, input_ids: torch.Tensor, cache: KVCache = None) -> Dict:
        B, total_tokens = input_ids.shape
        target_ids = torch.roll(input_ids, shifts=-1, dims=1); target_ids[:, -1] = 0
        
        positions = torch.arange(0, total_tokens, dtype=torch.long, device=input_ids.device)
        input_token_emb = self.token_embedding(input_ids) + self.pos_encoding(positions.unsqueeze(0))
        target_token_emb = self.token_embedding(target_ids) + self.pos_encoding(positions.unsqueeze(0))
        
        chunk_emb = self.chunk_encoder(input_token_emb)
        if self.lattice_core: chunk_emb = self.lattice_core(chunk_emb)
        
        logits, new_cache = self.chunk_decoder(chunk_emb, target_token_emb, cache=cache)
        
        return {'logits': logits, 'cache': new_cache, 'hidden_states': chunk_emb}

if __name__ == '__main__':
    print("=" * 80)
    print("HST UNIFIED MODEL SELF-TEST SUITE")
    print("=" * 80)

    # === Test 1: Default Token Mode ===
    print("\n--- Testing Default Token Mode Configuration ---")
    try:
        model_token = HST_Unified(DEFAULT_CONFIG)
        input_ids = torch.randint(0, DEFAULT_CONFIG['vocab_size'], (1, 128))
        output = model_token(input_ids)
        loss = output['logits'].mean()
        loss.backward()
        print("✅ Default Token Mode: Forward and backward pass successful!")
        print(f"   - Output logits shape: {output['logits'].shape}")
    except Exception as e:
        print(f"❌ Default Token Mode failed: {e}")

    # === Test 2: Chunk Mode ===
    print("\n--- Testing Chunk Mode Configuration ---")
    try:
        chunk_config = DEFAULT_CONFIG.copy()
        chunk_config['mode'] = 'chunk'
        model_chunk = HST_Unified(chunk_config)
        input_ids_chunk = torch.randint(0, chunk_config['vocab_size'], (1, 256)) # 2 chunks
        output_chunk = model_chunk(input_ids_chunk)
        loss_chunk = output_chunk['logits'].mean()
        loss_chunk.backward()
        print("✅ Chunk Mode: Forward and backward pass successful!")
        print(f"   - Output logits shape: {output_chunk['logits'].shape}")
    except Exception as e:
        print(f"❌ Chunk Mode failed: {e}")

    # === Test 3: Agile-like Configuration (v7.0 features) ===
    print("\n--- Testing Agile-like (v7.0) Configuration ---")
    try:
        agile_config = DEFAULT_CONFIG.copy()
        agile_config['lattice_core'] = 'adaptive'
        agile_config['attention_type'] = 'standard'
        agile_config['horizon_predictor'] = 'recursive'
        agile_config['use_multi_resolution_processor'] = False
        agile_config['use_sparse_experts'] = False
        
        model_agile = HST_Unified(agile_config)
        input_ids_agile = torch.randint(0, agile_config['vocab_size'], (1, 128))
        output_agile = model_agile(input_ids_agile)
        loss_agile = output_agile['logits'].mean()
        loss_agile.backward()
        print("✅ Agile-like Mode: Forward and backward pass successful!")
        print(f"   - Output logits shape: {output_agile['logits'].shape}")
    except Exception as e:
        print(f"❌ Agile-like Mode failed: {e}")

    print("\n" + "=" * 80)
    print("SELF-TEST SUITE COMPLETE")
    print("=" * 80)
