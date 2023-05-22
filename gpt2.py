#%%
import torch as t
from torch import nn
from einops import rearrange, einsum
from typing import Optional
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

class Attention(nn.Module):
    def __init__(self, n_embd: int, n_heads: int, head_embd: Optional[int] = None):
        super().__init__()
        assert n_embd % n_heads == 0
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.head_embd = n_embd // n_heads if head_embd is None else head_embd
        self.total_head_embed = self.head_embd * self.n_heads
        self.qkv_proj = nn.Linear(n_embd, 3 * self.total_head_embed)
        self.output_proj = nn.Linear(self.total_head_embed, n_embd)
        self.attn_pattern = nn.Identity()

    def forward(self, x: t.Tensor) -> t.Tensor:
        """(batch, seq, n_embd) -> (batch, seq, n_embd)"""
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, "b s (t h e) -> t b h s e", t=3, h=self.n_heads)
        q_k = (q @ k.transpose(-2, -1)) / (self.head_embd ** 0.5)
        mask = t.full((q_k.shape[-1], q_k.shape[-1]), -1e4).triu(1)
        attn = nn.functional.softmax(q_k.tril() + mask, dim=-1)
        attn = self.attn_pattern(attn)
        z = einsum(attn, v, "b h r c, b h c e -> b h r e")
        z = rearrange(z, "b h s e -> b s (h e)")
        return self.output_proj(z), attn

class GPT2Block(nn.Module):
    def __init__(self, n_embd: int, n_heads: int, ln_eps: float):
        super().__init__()
        self.ln1 = nn.LayerNorm((n_embd), eps=ln_eps)
        self.attn = Attention(n_embd, n_heads, None)
        self.ln2 = nn.LayerNorm((n_embd), eps=ln_eps)
        self.linear1 = nn.Linear(n_embd, 4*n_embd)
        self.linear2 = nn.Linear(4*n_embd, n_embd)
        self.mlp_act = nn.Identity()
        self.input_resid = nn.Identity()
        self.final_resid = nn.Identity()

    def forward(self, x: t.Tensor, patch_mlp_act=None) -> t.Tensor:
        """(batch, seq, n_embd) -> (batch, seq, n_embd)"""
        x = self.input_resid(x)
        attn_out, attn = self.attn(self.ln1(x))
        mid = x + attn_out
        mlp_act = nn.functional.gelu(self.linear1(self.ln2(mid)), approximate="tanh")
        mlp_act = mlp_act if patch_mlp_act is None else patch_mlp_act
        mlp_act = self.mlp_act(mlp_act)
        mlp_out = self.linear2(mlp_act)
        final_resid = self.final_resid(mid + mlp_out)
        return final_resid, attn, attn_out, mid, mlp_act, mlp_out

class GPT2(nn.Module):
    def __init__(self, vocab_size, max_pos, n_embd, n_layers, n_heads, ln_eps):
        super().__init__()
        self.vocab_size, self.max_pos, self.n_embd, self.n_layers, self.n_heads, self.ln_eps = \
            vocab_size, max_pos, n_embd, n_layers, n_heads, ln_eps
        self.tok_embed = nn.Embedding(vocab_size, n_embd)
        self.pos_embed = nn.Embedding(max_pos, n_embd)
        self.blocks = nn.Sequential(
            *[GPT2Block(n_embd, n_heads, ln_eps) for _ in range(n_layers)])
        self.final_ln = nn.LayerNorm((n_embd), ln_eps)

    def forward(self, x:t.Tensor, in_lyr:int=0, pos_embd=True, mlp_act_lyr=None, mlp_act=None) -> t.Tensor:
        """(batch, seq, [embed]), int64 -> (batch, seq, vocab_size), float32"""
        pos_vector = t.stack([t.arange(x.shape[1]) for _ in range(x.shape[0])])
        x = self.tok_embed(x) if x.ndim==2 else x
        x = (x + self.pos_embed(pos_vector)) if pos_embd else x
        act_lists = [[x], [], [], [], [], []]
        for i, block in enumerate(self.blocks[in_lyr:]):
            block_outs = block(x, mlp_act if i == mlp_act_lyr else None)
            x = block_outs[0]
            [l.append(o) for l, o in zip(act_lists, block_outs)]
        x = self.final_ln(x)
        acts = [t.stack(l) for l in act_lists if len(l) > 0]
        return einsum(x, self.tok_embed.weight, "b s e, v e -> b s v"), acts

def _copy_weight_bias(mine, theirs, T=False):
    mine.weight.data.copy_(theirs.weight.T if T else theirs.weight)
    if mine.bias is not None:
        mine.bias.data.copy_(theirs.bias)

def load_pretrained_gpt2(name: str = "gpt2", device: str = "cpu"):
    # Load from cache if it exists
    cache_path = Path(f".cache/{name}.pt")
    cache_path.parent.mkdir(exist_ok=True)

    if name == "gpt2": my_gpt2 = GPT2(50257, 1024, 768, 12, 12, 1e-5)
    elif name == "gpt2-medium": my_gpt2 = GPT2(50257, 1024, 1024, 24, 16, 1e-5)
    elif name == "gpt2-large": my_gpt2 = GPT2(50257, 1024, 1280, 36, 20, 1e-5)
    elif name == "gpt2-xl": my_gpt2 = GPT2(50257, 1024, 1600, 48, 25, 1e-5)
    else: raise ValueError(f"Unknown GPT2 model {name}")

    if os.path.exists(cache_path):
        print(f"Loading {name} from cache")
        my_gpt2.load_state_dict(t.load(cache_path))
        return my_gpt2.to(device), AutoTokenizer.from_pretrained(name)

    hf_gpt2 = AutoModelForCausalLM.from_pretrained(name)
    my_gpt2.tok_embed.weight.data.copy_(hf_gpt2.transformer.wte.weight)
    my_gpt2.pos_embed.weight.data.copy_(hf_gpt2.transformer.wpe.weight)
    _copy_weight_bias(my_gpt2.final_ln, hf_gpt2.transformer.ln_f)
    for (my_block, hf_block) in zip(my_gpt2.blocks, hf_gpt2.transformer.h):
        _copy_weight_bias(my_block.ln1, hf_block.ln_1)
        _copy_weight_bias(my_block.attn.qkv_proj, hf_block.attn.c_attn, T=True)
        _copy_weight_bias(my_block.attn.output_proj, hf_block.attn.c_proj, T=True)
        _copy_weight_bias(my_block.ln2, hf_block.ln_2)
        _copy_weight_bias(my_block.linear1, hf_block.mlp.c_fc, T=True)
        _copy_weight_bias(my_block.linear2, hf_block.mlp.c_proj, T=True)
    t.save(my_gpt2.state_dict(), cache_path)
    return my_gpt2.to(device), AutoTokenizer.from_pretrained(name)

#%%
# Test implementation matches HuggingFace
if __name__ == "__main__":
    model_name = "gpt2-medium"
    my_gpt, my_tknizr = load_pretrained_gpt2(model_name)
    hf_tknizr = AutoTokenizer.from_pretrained(model_name)
    hf_gpt2 = AutoModelForCausalLM.from_pretrained(model_name)
    encode = lambda text: hf_tknizr(text, return_tensors="pt")["input_ids"]
    tokens = encode("I like to visit the Colosseum when I go to")
    with t.inference_mode():
        logits, hf_logits = my_gpt(tokens)[0][0, -1], hf_gpt2(tokens)[0][0, -1]
    topk, hf_topk = t.topk(logits, k=10), t.topk(hf_logits, k=10)
    for i in range(10):
        print("Rank", i)
        print("Mine:", f"{my_tknizr.decode(topk.indices[i])} {topk.values[i]}")
        print("HF:", f"{hf_tknizr.decode(hf_topk.indices[i])} {hf_topk.values[i]}")