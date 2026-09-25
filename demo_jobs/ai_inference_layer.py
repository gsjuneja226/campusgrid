"""
CampusGrid: Distributed AI Inference Worker
Loads a specific layer shard of GPT-2 and performs a forward pass.
Used in pipeline-parallel inference: hidden states flow through workers sequentially.

This module is called by master.py's AI inference coordinator,
NOT through the standard chunk scheduler (inference is sequential, not parallel).
"""
import os
import time
import base64
import struct
import numpy as np

# Lazy-load torch/transformers to avoid import errors on workers that don't have them
_gpt2_model = None
_gpt2_tokenizer = None
_loaded_layer_range = None


def _load_model_shard(layer_start: int, layer_end: int):
    """
    Loads only layers [layer_start, layer_end) of GPT-2 into memory.
    Caches the result so repeated calls don't reload.
    """
    global _gpt2_model, _gpt2_tokenizer, _loaded_layer_range

    if _loaded_layer_range == (layer_start, layer_end):
        return _gpt2_model, _gpt2_tokenizer  # already loaded

    import torch
    from transformers import GPT2Tokenizer, GPT2Model, GPT2Config

    print(f"[AI] Loading GPT-2 layer shard [{layer_start}, {layer_end})...")
    t0 = time.time()

    # Download/load full GPT-2 config + tokenizer (fast)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    config = GPT2Config.from_pretrained("gpt2")

    # Load full model weights then extract only our layer shard
    full_model = GPT2Model.from_pretrained("gpt2")
    full_model.eval()

    # Build a shard wrapper
    class GPT2LayerShard(torch.nn.Module):
        def __init__(self, model, layer_start, layer_end):
            super().__init__()
            self.wte = model.wte   # token embeddings (only used at layer 0)
            self.wpe = model.wpe   # position embeddings
            self.drop = model.drop
            self.layers = torch.nn.ModuleList(model.h[layer_start:layer_end])
            self.ln_f = model.ln_f  # final layer norm (only used at last shard)
            self.layer_start = layer_start
            self.layer_end = layer_end
            self.num_layers = config.n_layer

        def forward(self, input_ids=None, hidden_states=None, position_ids=None):
            """
            If layer_start == 0: expects input_ids, computes initial embeddings.
            Otherwise: expects hidden_states tensor directly.
            If layer_end == num_layers: applies final ln_f.
            """
            import torch
            if self.layer_start == 0:
                # First shard: compute embeddings
                device = next(self.parameters()).device
                if position_ids is None:
                    seq_len = input_ids.shape[-1]
                    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                hidden_states = self.wte(input_ids) + self.wpe(position_ids)
                hidden_states = self.drop(hidden_states)

            for layer in self.layers:
                outputs = layer(hidden_states)
                hidden_states = outputs[0]

            if self.layer_end == self.num_layers:
                hidden_states = self.ln_f(hidden_states)

            return hidden_states

    shard = GPT2LayerShard(full_model, layer_start, layer_end)
    shard.eval()

    # Free the full model from memory (keep only shard)
    del full_model

    _gpt2_model = shard
    _gpt2_tokenizer = tokenizer
    _loaded_layer_range = (layer_start, layer_end)

    print(f"[AI] Layer shard [{layer_start},{layer_end}) loaded in {time.time()-t0:.1f}s")
    return shard, tokenizer


def _tensor_to_b64(tensor) -> str:
    """Serialize a torch tensor to base64 string via numpy."""
    arr = tensor.detach().cpu().numpy()
    buf = arr.tobytes()
    meta = struct.pack("!I" + "I" * len(arr.shape), len(arr.shape), *arr.shape)
    dtype_str = str(arr.dtype).encode("utf-8").ljust(8)[:8]
    return base64.b64encode(dtype_str + meta + buf).decode("ascii")


def _b64_to_tensor(b64_str: str):
    """Deserialize a base64 string back to a torch tensor."""
    import torch
    data = base64.b64decode(b64_str)
    dtype_str = data[:8].strip().decode("utf-8")
    dtype_map = {
        "float32": np.float32, "float16": np.float16, "float64": np.float64
    }
    np_dtype = dtype_map.get(dtype_str, np.float32)
    meta_start = 8
    ndim = struct.unpack("!I", data[meta_start:meta_start+4])[0]
    shape = struct.unpack("!" + "I" * ndim, data[meta_start+4:meta_start+4+4*ndim])
    buf_start = meta_start + 4 + 4 * ndim
    arr = np.frombuffer(data[buf_start:], dtype=np_dtype).reshape(shape).copy()
    return torch.from_numpy(arr)


def run_layer_shard(layer_start: int, layer_end: int, input_b64: str,
                    input_ids_b64: str = None, is_first: bool = False) -> str:
    """
    Performs a forward pass through layers [layer_start, layer_end).

    Args:
        layer_start    : first layer index (0-indexed)
        layer_end      : last layer index (exclusive)
        input_b64      : base64 hidden states tensor (used if not first shard)
        input_ids_b64  : base64 token ids tensor (used only for first shard)
        is_first       : True if this is the first shard (uses token embeddings)

    Returns:
        base64 hidden states tensor (output of this shard)
    """
    import torch

    shard, tokenizer = _load_model_shard(layer_start, layer_end)

    with torch.no_grad():
        if is_first:
            input_ids = _b64_to_tensor(input_ids_b64).long()
            hidden_states = shard(input_ids=input_ids)
        else:
            hidden_states = _b64_to_tensor(input_b64)
            hidden_states = shard(hidden_states=hidden_states)

    return _tensor_to_b64(hidden_states)


def get_vocab_logits(hidden_states_b64: str) -> list:
    """
    Decodes final hidden states to next-token probabilities using GPT-2's LM head.
    Returns top-k token ids and probabilities.
    """
    import torch
    from transformers import GPT2LMHeadModel

    hidden_states = _b64_to_tensor(hidden_states_b64)

    # Load LM head (just the final linear projection, tiny)
    lm_model = GPT2LMHeadModel.from_pretrained("gpt2")
    lm_model.eval()

    with torch.no_grad():
        logits = lm_model.lm_head(hidden_states)  # shape: [1, seq_len, vocab_size]
        next_token_logits = logits[0, -1, :]       # last token position
        probs = torch.softmax(next_token_logits, dim=-1)
        top_k = torch.topk(probs, k=10)

    return {
        "top_ids":   top_k.indices.tolist(),
        "top_probs": top_k.values.tolist()
    }
