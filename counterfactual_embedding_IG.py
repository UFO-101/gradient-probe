#%%
import torch as t
import importlib, gpt2, utils
from einops import rearrange
importlib.reload(gpt2)
import pandas as pd
importlib.reload(utils)
import plotly.express as px
from gpt2 import load_pretrained_gpt2
from utils import print_preds, logit_diff, display_tensor_grid, tokenize, logit, tokenize_txt

device = t.device("cuda" if t.cuda.is_available() else "cpu")
model, tknizr = load_pretrained_gpt2("gpt2-large", device)


a, an = " Germany", " France"
prompt = ' city is the capital of'
fruit_idx = 0
prompt_toks = tokenize(prompt, tknizr, device=device)[0]
lemon_tok = tokenize(" Berlin", tknizr, device=device)[0]
apple_tok = tokenize(" Paris", tknizr, device=device)[0]

prompt_embd = model.tok_embed(prompt_toks)
lemon_embd, apple_embd = model.tok_embed(lemon_tok), model.tok_embed(apple_tok)

class FruitPromptEmbd(t.nn.Module):
    def __init__(self, prompt_embd, fruit_idx):
        super().__init__()
        self.prompt_embd = prompt_embd.detach().clone()
        self.fruit_idx = fruit_idx
    def forward(self, fruit_embd):
        x = self.prompt_embd.clone()
        x[:, self.fruit_idx, :] = fruit_embd
        return x
fruit_prompt_embd = FruitPromptEmbd(prompt_embd, fruit_idx)

output_0, _ = model(fruit_prompt_embd(apple_embd), pos_embd=True)
output_1, _ = model(fruit_prompt_embd(lemon_embd), pos_embd=True)
baseline_diff = logit_diff(an, a, output_0, tknizr) \
                - logit_diff(an, a, output_1, tknizr)

print("Output 0")
print_preds(output_0, tknizr)
print("Output 1")
print_preds(output_1, tknizr)

samples = 1000
integrated_grads = t.zeros_like(lemon_embd)
prev_loss = None
prev_grad = None
for i in range(0, samples + 1):
    model.zero_grad()
    lerp_embd = (lemon_embd + ((i / samples) * (apple_embd - lemon_embd))).detach()
    lerp_embd.requires_grad = True
    lerp_output, _ = model(fruit_prompt_embd(lerp_embd), pos_embd=True)
    loss = logit_diff(an, a, lerp_output, tknizr)
    loss.backward()

    if prev_loss is not None and prev_grad is not None:
        loss_diff = loss - prev_loss
        integrated_grads += (prev_grad**2 *loss_diff)/t.sum(prev_grad**2)

    prev_loss = loss
    prev_grad = lerp_embd.grad

print("Baseline diff:", baseline_diff.item())
print("Integrated grads:", integrated_grads.sum(dim=(0, 1, 2)).item())

#%%
print(integrated_grads.shape)
px.scatter(integrated_grads[0, 0].detach().cpu())
#%%
ig_dir = integrated_grads[0, 0].detach()
lemon_dir, apple_dir = lemon_embd[0, 0].detach(), apple_embd[0, 0].detach()
# a_an_dir = ig_dir * (apple_dir - lemon_dir)
a_an_dir = apple_dir - lemon_dir
a_an_dir = a_an_dir / a_an_dir.norm()

# nouns = ' lemon apple pear orange banana'
nouns = ' Berlin Paris London Rome Madrid Lyon Munich Hamburg Nice France French'
noun_toks = tokenize(nouns, tknizr, device=device)[0]
noun_embd = model.tok_embed(noun_toks)
print('noun embd', noun_embd.shape, 'a_an_dir', a_an_dir.shape)
noun_projs = ((noun_embd[0] @ a_an_dir) / (a_an_dir.norm()))
print(noun_projs.shape)
px.bar(y=noun_projs.detach().cpu(), x=nouns[1:].split(' '))
# %%

lemon_proj = ((lemon_dir @ a_an_dir) / (a_an_dir @ a_an_dir)) * a_an_dir
apple_proj = ((apple_dir @ a_an_dir) / (a_an_dir @ a_an_dir)) * a_an_dir
an_lemon_dir = lemon_dir + (apple_proj - lemon_proj)
an_lemon_embd = an_lemon_dir.unsqueeze(0).unsqueeze(0)
an_lemon_output, _ = model(fruit_prompt_embd(an_lemon_embd), pos_embd=True)
print_preds(an_lemon_output, tknizr)

#%%
# validate_prompt = ' The most famous landmark in city is the'
validate_prompt = ' city is the leader of'
validate_prompt_toks = tokenize(validate_prompt, tknizr, device=device)[0]
validate_prompt_embd = model.tok_embed(validate_prompt_toks)
# validate_idx = 5
validate_idx = 0
validate_prompt_embd = FruitPromptEmbd(validate_prompt_embd, validate_idx)

london_tok = tokenize(" Merkel", tknizr, device=device)[0]
london_embd = model.tok_embed(london_tok)
validate_london_output, _ = model(validate_prompt_embd(london_embd), pos_embd=True)
print() or print("Validate London output")
print_preds(validate_london_output, tknizr)
print("France logit:", logit(an, validate_london_output, tknizr).item())

london_proj = ((london_embd[0, 0] @ a_an_dir) / (a_an_dir @ a_an_dir)) * a_an_dir
french_london_embd = (london_embd[0, 0] + (apple_proj - london_proj)).unsqueeze(0).unsqueeze(0)
french_london_output, _ = model(validate_prompt_embd(french_london_embd), pos_embd=True)
print() or print("Validate French London output")
print_preds(french_london_output, tknizr)
print("Grance logit:", logit(an, french_london_output, tknizr).item())

#%%
# validate_apple_output, _ = model(validate_prompt_embd(apple_embd), pos_embd=True)
# print() or print("Validate apple (Paris) output")
# print_preds(validate_apple_output, tknizr)
# validate_lemon_output, _ = model(validate_prompt_embd(lemon_embd), pos_embd=True)
# print() or print("Validate lemon (Berlin) output")
# print_preds(validate_lemon_output, tknizr)
validate_an_lemon_output, _ = model(validate_prompt_embd(an_lemon_embd), pos_embd=True)
print() or print("Validate AN (Berlin) lemon output")
print_preds(validate_an_lemon_output, tknizr)