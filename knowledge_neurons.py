#%%
import torch as t
import importlib, gpt2, utils
from einops import rearrange
from functools import partial
importlib.reload(gpt2)
import pandas as pd
importlib.reload(utils)
import plotly.express as px
from gpt2 import load_pretrained_gpt2
from utils import print_preds, logit_diff, display_tensor_grid, tokenize, logit

COUNTERFACTUAL = False
LAST_TOKEN = False

device = t.device("cuda" if t.cuda.is_available() else "cpu")
model, tknizr = load_pretrained_gpt2("gpt2-xl", device)

# prompt_0 = "I climbed up the pear tree and picked a pear. I climbed up the lemon tree and picked"
# prompt_1 = "I climbed up the pear tree and picked a pear. I climbed up the apple tree and picked"
prompt_0 = "blah blah"
prompt_1 = "Germaine Greer's domain of work is"
# a_tok, an_tok = " a", " an"
feminism_tok = " feminism"
tokens_0 = tokenize(prompt_0, tknizr)[0].to(device)
tokens_1 = tokenize(prompt_1, tknizr)[0].to(device)

if COUNTERFACTUAL:
    output_0, (_, _, _, _, mlp_acts_0, _) = model(tokens_0)
else:
    embed_1 = model.tok_embed(tokens_1)
    output_0, (_, _, _, _, mlp_acts_0, _) = model(t.zeros_like(embed_1), pos_embd=True)
output_1, (resids_1, _, _, _, mlp_acts_1, _) = model(tokens_1)
# baseline_diff = logit_diff(an_tok, a_tok, output_1, tknizr) \
                # - logit_diff(an_tok, a_tok, output_0, tknizr)
baseline_diff = logit(feminism_tok, output_1, tknizr) \
                - logit(feminism_tok, output_0, tknizr)

print("Output 0") or print_preds(output_0, tknizr)
print("Output 1") or print_preds(output_1, tknizr)

class MlpActs(t.nn.Module):
    def __init__(self, default_acts):
        super().__init__()
        self.default_acts = default_acts.detach()
    def forward(self, lerp_acts, lyr):
        if len(lerp_acts.shape) < 3:
            patch = self.default_acts.detach().clone()
            patch[:, -1] = lerp_acts
        else:
            patch = lerp_acts
        return model(tokens_1, pos_embd=True, mlp_act_lyr=lyr, mlp_act=patch)

samples = 50
integrated_grads = t.zeros_like(mlp_acts_1[:, :, -1] if LAST_TOKEN else mlp_acts_1)
# for lyr in range(0, mlp_acts_0.shape[0]):
for lyr in range(0, mlp_acts_1.shape[0]):
    prev_loss = None
    prev_grad = None
    parse_act = MlpActs(mlp_acts_1[lyr])
    act_0 = mlp_acts_0[lyr, :, -1] if LAST_TOKEN else mlp_acts_0[lyr]
    act_1 = mlp_acts_1[lyr, :, -1] if LAST_TOKEN else mlp_acts_1[lyr]
    act_0 = act_0 if COUNTERFACTUAL else t.zeros_like(act_1)

    for i in range(0, samples + 1):
        parse_act.zero_grad()
        lerp_act = (act_0 + (i / samples) * (act_1 - act_0)).detach()
        lerp_act.requires_grad = True
        lerp_output, _ = parse_act(lerp_act, lyr)
        # loss = logit_diff(an_tok, a_tok, lerp_output, tknizr)
        loss = logit(feminism_tok, lerp_output, tknizr)
        loss.backward()

        if prev_loss is not None and prev_grad is not None:
            loss_diff = loss - prev_loss
            integrated_grads[lyr] += (prev_grad**2 *loss_diff)/t.sum(prev_grad**2)

        prev_loss = loss
        prev_grad = lerp_act.grad

print("Counterfactual:", COUNTERFACTUAL, "Last token only:", LAST_TOKEN)
print("Baseline diff:", baseline_diff)
print("Integrated grads:", integrated_grads.sum(dim=(1, 2)))
#%%

#%%
px.imshow(integrated_grads[:, 0].norm(dim=-1).T.detach().cpu()).show()