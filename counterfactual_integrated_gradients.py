#%%
import torch as t
import importlib, gpt2, utils
from einops import rearrange
importlib.reload(gpt2)
import pandas as pd
importlib.reload(utils)
import plotly.express as px
from gpt2 import load_pretrained_gpt2
from utils import print_preds, logit_diff, display_tensor_grid, tokenize, logit
import time

start = time.time()
COUNTERFACTUAL = True
TKN_POS = None

device = t.device("cuda" if t.cuda.is_available() else "cpu")
model, tknizr = load_pretrained_gpt2("gpt2-large", device)

prompt_0 = ' Berlin is the capital of'
prompt_1 = ' Paris is the capital of'
france_tok, germany_tok = " France", " Germany"
tokens_0 = tokenize(prompt_0, tknizr)[0].to(device)
tokens_1 = tokenize(prompt_1, tknizr)[0].to(device)

output_0, (resids_0, _, _, _, _, _) = model(tokens_0)
output_1, (resids_1, _, _, _, _, _) = model(tokens_1)

print("Output 0", "France logit:", logit(" France", output_0, tknizr), "Germany logit:", logit(" Germany", output_0, tknizr))
print("Output 1", "France logit:", logit(" France", output_1, tknizr), "Germany logit:", logit(" Germany", output_1, tknizr))

#%%
# print("Output 0")
# print_preds(output_0, tknizr)
# print("Output 1")
# print_preds(output_1, tknizr)

class PatchResid(t.nn.Module):
    def __init__(self, default_acts):
        super().__init__()
        self.default_acts = default_acts.detach()
    def forward(self, lerp_resid, lyr, tkn_pos = None):
        if tkn_pos is not None:
            patch = self.default_acts.detach().clone()
            patch[:, tkn_pos] = lerp_resid
        else:
            patch = lerp_resid
        return model(patch, in_lyr=lyr, pos_embd=False)

samples = 100
baseline_diffs = []
integrated_grads = t.zeros_like(resids_0)
for lyr in range(0, resids_1.shape[0]):
# for lyr in range(5, 6):
    prev_loss = None
    prev_grad = None
    patch_resid = PatchResid(resids_1[lyr])
    resid_0 = resids_0[lyr] if TKN_POS is None else resids_0[lyr, :, TKN_POS]
    resid_1 = resids_1[lyr] if TKN_POS is None else resids_1[lyr, :, TKN_POS]
    resid_0 = resid_0 if COUNTERFACTUAL else t.zeros_like(resid_1)
    patched_output, _ = patch_resid(resid_0, lyr=lyr, tkn_pos=TKN_POS)
    baseline_diffs.append(logit_diff(france_tok, germany_tok, patched_output, tknizr).item())
    # for i in range(0, samples + 1):
    #     patch_resid.zero_grad()
    #     lerp_resid = (resid_0 + ((i / samples) * (resid_1 - resid_0))).detach()
    #     lerp_resid.requires_grad = True
    #     lerp_output, _ = patch_resid(lerp_resid, lyr=lyr, tkn_pos=TKN_POS)
    #     loss = logit_diff(france_tok, germany_tok, lerp_output, tknizr)
    #     loss.backward()

    #     if prev_loss is not None and prev_grad is not None:
    #         loss_diff = loss - prev_loss
    #         integrated_grads[lyr] += (prev_grad**2 *loss_diff)/t.sum(prev_grad**2)

    #     prev_loss = loss
    #     prev_grad = lerp_resid.grad

print("time elapsed:", time.time() - start)
print("Baseline diff:", baseline_diffs)
print("Integrated grads:", integrated_grads.sum(dim=(1, 2, 3)))
#%%
px.scatter(integrated_grads[5, 0].T.detach().cpu()).show()
# px.scatter((embed_1 - embed_0)[0].T.detach().cpu()).show()
#%%

test_prompt = " Berlin is the capital of"
test_tokens = tokenize(test_prompt, tknizr)[0].to(device)
test_embed = model.tok_embed(test_tokens)
test_output, (test_resids, _, _, _, _, _) = model(test_tokens, pos_embd=True)

