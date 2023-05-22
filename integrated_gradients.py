#%%
import torch as t
import importlib, gpt2, utils
from functools import partial
importlib.reload(gpt2)
importlib.reload(utils)
import plotly.express as px
from gpt2 import load_pretrained_gpt2
from utils import print_preds, logit_diff, display_tensor_grid, tokenize, logit, tokenize_txt

device = t.device("cuda" if t.cuda.is_available() else "cpu")
model, tknizr = load_pretrained_gpt2("gpt2", device)

clean_prompt = 'When John and Mary went to the shops, John gave the bag to'
corrupt_prompt = 'When John and Mary went to the shops, Mary gave the bag to'
right_tok, wrong_tok = " Mary", " John"
clean_tokens = tokenize(clean_prompt, tknizr).to(device)
corrupt_tokens = tokenize(corrupt_prompt, tknizr).to(device)

clean_embed = model.tok_embed(clean_tokens)
corrupt_embed = model.tok_embed(corrupt_tokens)
clean_output, (clean_resids, clean_attn, _, _, clean_mlp_act, _) = model(clean_tokens)
corrupt_output, (corrupt_resids, corrupt_attn, _, _, corrupt_mlp_act, _) = model(corrupt_tokens)

samples = 500
# grad_sum = t.zeros_like(clean_attn)
# grad_sum = t.zeros_like(clean_mlp_act)
# grad_sum = t.zeros_like(clean_embed)
grad_sum = t.zeros_like(clean_resids)
def grad_sum_hook(module, grad_in, grad_out, layer):
    global grad_sum
    grad_sum[layer] += grad_out[0]

for i in range(1, samples + 1):
    lerp_embed = corrupt_embed + ((i / samples) * (clean_embed - corrupt_embed))
    lerp_embed = lerp_embed.detach()
    lerp_embed.requires_grad = True
    try:
        handles = []
        for layer, block in enumerate(model.blocks):
            hook = partial(grad_sum_hook, layer=layer)
            # handles.append(block.attn.attn_pattern.register_full_backward_hook(hook))
            # handles.append(block.mlp_act.register_full_backward_hook(hook))
            handles.append(block.input_resid.register_full_backward_hook(hook))
        lerp_output, _ = model(lerp_embed, embed=False)
    finally:
        for handle in handles:
            handle.remove()
    loss = logit_diff(right_tok, wrong_tok, lerp_output, tknizr)
    loss.backward()
    # grad_sum += lerp_embed.grad

baseline_diff = logit_diff(right_tok, wrong_tok, clean_output, tknizr) - logit_diff(right_tok, wrong_tok, corrupt_output, tknizr)
print("Baseline diff:", baseline_diff)

# integrated_grad = (clean_attn - corrupt_attn) * grad_sum / samples
# integrated_grad = (clean_mlp_act - corrupt_mlp_act) * grad_sum / samples
integrated_grad = (clean_resids - corrupt_resids) * grad_sum / samples
# integrated_grad = (clean_embed - corrupt_embed) * grad_sum / samples
print('integrated_grad', integrated_grad.shape)
# integrated_grad_sum = t.tril(integrated_grad).sum()
# print("Integrated grad sum:", integrated_grad_sum)
# print("t.tril(integrated_grad).sum()", t.tril(integrated_grad).sum())
print("integrated_grad.sum()", integrated_grad.sum())
# print("t.tril(grad_sum / samples).sum()", t.tril(grad_sum / samples).sum())
# mask = t.full_like(grad_sum, -1).triu(1)
# integrated_grad = t.tril(integrated_grad) # + mask
# display_tensor_grid(integrated_grad, "Integrated Grad", True, tokenize_txt(clean_prompt, tknizr))
# print("No gradients!!!!!!!")
# display_tensor_grid(t.tril(clean_attn-corrupt_attn), "Integrated Grad", True, tokenize_txt(clean_prompt, tknizr))
# px.line(integrated_grad[0].T.detach())
# px.line(integrated_grad[:, 0, -1].T.detach())
# px.line(integrated_grad[:, 0, -2].T.detach())

#%%
print("integrated_grad sum", integrated_grad[0, 0, :].sum())
px.line(integrated_grad[0, 0, :].T.detach())