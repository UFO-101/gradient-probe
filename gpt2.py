import torch as t
from torch import nn
from einops import rearrange, einsum
from typing import Optional
import transformers

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

    def forward(self, x: t.Tensor) -> t.Tensor:
        """(batch, seq, n_embd) -> (batch, seq, n_embd)"""
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, "b s (t h e) -> t b h s e", t=3, h=self.n_heads)
        q_k = (q @ k.transpose(-2, -1)) / (self.head_embd ** 0.5)
        mask = t.full((q_k.shape[-1], q_k.shape[-1]), -1e4).triu(1)
        attn = nn.functional.softmax(q_k.tril() + mask, dim=-1)
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

    def forward(self, x: t.Tensor) -> t.Tensor:
        """(batch, seq, n_embd) -> (batch, seq, n_embd)"""
        attn_out, attn = self.attn(self.ln1(x))
        mid = x + attn_out
        mlp_act = nn.functional.gelu(self.linear1(self.ln2(mid)), approximate="tanh")
        mlp_out = self.linear2(mlp_act)
        return mid + mlp_out, attn, attn_out, mid, mlp_act, mlp_out

class GPT2(nn.Module):
    def __init__(self, vocab_size, max_pos, n_embd, n_layers, n_heads, ln_eps):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, n_embd)
        self.pos_embed = nn.Embedding(max_pos, n_embd)
        self.blocks = nn.Sequential(
            *[GPT2Block(n_embd, n_heads, ln_eps) for _ in range(n_layers)])
        self.final_ln = nn.LayerNorm((n_embd), ln_eps)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """(batch, seq), int64 -> (batch, seq, vocab_size), float32"""
        pos_vector = t.stack([t.arange(x.shape[1]) for _ in range(x.shape[0])])
        x = self.tok_embed(x) + self.pos_embed(pos_vector)
        act_lists = [[x], [], [], [], [], []]
        for block in self.blocks:
            block_outs = block(x)
            x = block_outs[0]
            [l.append(o) for l, o in zip(act_lists, block_outs)]
        x = self.final_ln(x)
        acts = [t.stack(l) for l in act_lists]
        return einsum(x, self.tok_embed.weight, "b s e, v e -> b s v"), acts

def _copy_weight_bias(mine, theirs, transpose=False):
    mine.weight.data.copy_(theirs.weight.T if transpose else theirs.weight)
    if mine.bias is not None:
        mine.bias.data.copy_(theirs.bias)

def load_pretrained_gpt2():
    hf_gpt2 = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    my_gpt2 = GPT2(50257, 1024, 768, 12, 12, 1e-5)
    my_gpt2.tok_embed.weight.data.copy_(hf_gpt2.transformer.wte.weight)
    my_gpt2.pos_embed.weight.data.copy_(hf_gpt2.transformer.wpe.weight)
    _copy_weight_bias(my_gpt2.final_ln, hf_gpt2.transformer.ln_f)

    from transformers.models.gpt2.modeling_gpt2 import GPT2Block as HFGPT2Block
    my_block: GPT2Block
    hf_block: HFGPT2Block
    for (my_block, hf_block) in zip(my_gpt2.blocks, hf_gpt2.transformer.h):
        _copy_weight_bias(my_block.ln1, hf_block.ln_1)
        _copy_weight_bias(my_block.attn.qkv_proj, hf_block.attn.c_attn, transpose=True)
        _copy_weight_bias(my_block.attn.output_proj, hf_block.attn.c_proj, transpose=True)
        _copy_weight_bias(my_block.ln2, hf_block.ln_2)
        _copy_weight_bias(my_block.linear1, hf_block.mlp.c_fc, transpose=True)
        _copy_weight_bias(my_block.linear2, hf_block.mlp.c_proj, transpose=True)
    return my_gpt2

#%%
if __name__ == "__main__":
    # Test implementation matches HuggingFace
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    my_gpt = load_pretrained_gpt2()
    hf_gpt2 = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    def encode(text: str) -> t.Tensor:
            return tokenizer(text, return_tensors="pt")["input_ids"]
    tokens = encode("Former President of the United States of America, George")
    with t.inference_mode():
        logits, hf_logits = my_gpt(tokens)[0][0, -1], hf_gpt2(tokens)[0][0, -1]
    topk, hf_topk = t.topk(logits, k=10), t.topk(hf_logits, k=10)
    for i in range(10):
        print("Rank", i)
        print("Mine:", f"{tokenizer.decode(topk.indices[i])} {topk.values[i]}")
        print("HF:", f"{tokenizer.decode(hf_topk.indices[i])} {hf_topk.values[i]}")