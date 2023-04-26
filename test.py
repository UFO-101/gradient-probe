#%%
import torch as t
import importlib, gpt2, utils
import plotly.express as px
importlib.reload(gpt2)
importlib.reload(utils)
from gpt2 import load_pretrained_gpt2
from utils import print_preds, logit_diff, display_tensor_grid, tokenize, logit, tokenize_txt

device = t.device("cuda" if t.cuda.is_available() else "cpu")
model, tknizr = load_pretrained_gpt2("gpt2-large", device)

clean_prompt = "I climbed up the pear tree and picked a pear. I climbed up the apple tree and picked"
corrupted_prompt = "I climbed up the pear tree and picked a pear. I climbed up the lemon tree and picked"
right_tok, wrong_tok = " an", " a"
# clean_prompt = 'When John and Mary went to the shops, John gave the bag to'
# corrupted_prompt = 'When John and Mary went to the shops, Mary gave the bag to'
# right_tok, wrong_tok = " Mary", " John"

clean_tokens = tokenize(clean_prompt, tknizr).to(device)
print('clean toks', clean_tokens.shape)
corrupted_tokens = tokenize(corrupted_prompt, tknizr).to(device)

optimizer = t.optim.SGD(model.parameters(), lr=1e-3)
clean_output, (resid, attn, _, _, mlp_act, _) = model(clean_tokens)
print('attn', attn.shape)
corrupted_output, _ = model(corrupted_tokens)

clean_logit_diff = logit_diff(right_tok, wrong_tok, clean_output, tknizr)
corrupted_logit_diff = logit_diff(right_tok, wrong_tok, corrupted_output, tknizr)
loss = corrupted_logit_diff - clean_logit_diff
# loss = -clean_logit_diff
# loss = logit(wrong_tok, clean_output, tknizr) - logit(wrong_tok, corrupted_output, tknizr)
# loss = logit(wrong_tok, corrupted_output, tknizr) - logit(wrong_tok, clean_output, tknizr)
# loss = logit(right_tok, clean_output, tknizr) - logit(right_tok, corrupted_output, tknizr)

loss.backward()
optimizer.step()
mod_clean_output, (mod_resid, mod_attn, _, _, mod_mlp_act, _) = model(clean_tokens)
attn_diff = mod_attn - attn
mlp_diff = mod_mlp_act - mlp_act
# display_tensor_grid(-attn_diff, "Attention Difference", True, tokenize_txt(clean_prompt, tknizr))
print('mlp diff', mlp_diff.shape)
px.line(-mlp_diff[:, 0, -1].detach().T, title="MLP Difference")
