#%%
import torch as t
import importlib, gpt2, utils
from functools import partial
importlib.reload(gpt2)
importlib.reload(utils)
from einops import rearrange, einsum, repeat
import plotly.express as px
from gpt2 import load_pretrained_gpt2, GPT2
from utils import print_preds, logit_diff, display_tensor_grid, tokenize, logit, tokenize_txt
import time

start_time = time.time()

t.manual_seed(42)
device = t.device("cuda" if t.cuda.is_available() else "cpu")
trained_model, tknizr = load_pretrained_gpt2("gpt2-large", device)
base_model, _ = load_pretrained_gpt2("gpt2-large", device)
# Alter the base model parameters
weight_names = "linear1.weight"
trained_params = trained_model.state_dict()
base_params = base_model.state_dict()
for name, param in base_model.named_parameters():
    # if weight_names == "qkv_proj.weight" and weight_names in name:
    #     qkv_len = param.shape[0] // 3
    #     param.data[0: qkv_len] = param.data[0: qkv_len].mean()
    #     param.data[qkv_len: 2*qkv_len] = param.data[qkv_len: 2*qkv_len].mean()
    if weight_names == "linear1.weight" and weight_names in name:
        param.data[:, :] = param.data.mean()

lerp_model = GPT2(trained_model.vocab_size, trained_model.max_pos, trained_model.n_embd,
                  trained_model.n_layers, trained_model.n_heads, trained_model.ln_eps).to(device)

# prompt = 'When John and Mary went to the shops, John gave the bag to'
# corrupt_prompt = 'When John and Mary went to the shops, Mary gave the bag to'
# right_tok, wrong_tok = " Mary", " John"
# prompts = "I climbed up the pear tree and picked a pear. I climbed up the apple tree and picked"
prompts = ["Germaine Greer's domain of work is",
           "Germaine Greer was a proponent of",
           "Germaine Greer inspired me to learn about",
           "If we read Germaine Greer, we understand the movement of",
           "For example, Germaine Greer was a towering figure in",
           "He disliked Germaine Greer because he didn't agree with the ideology of",
           "Don't argue with Germaine Greer about",
           "Rosa Parks for civil rights, Helen Keller for disability activism, Germaine Greer for",
           "Thunberg is to environmental activism as Greer was to",
           "I can't stand speeches. I heard one by Germaine Greer on the topic of",
           "If you want to feel empowered as a woman, read Greer on"]
# a_tok, an_tok = " a", " an"
feminism_tok = " feminism"
# corrupt_prompt = "I climbed up the pear tree and picked a pear. I climbed up the lemon tree and picked"
# right_tok, wrong_tok = " an", " a"
# base_prompt = " The Eiffel Tower is located in the city of"
# prompt_prefixes = ["The U.S.", "The first thing.", "A few weeks ago.", "The New York Giants.", "In the early hours.", "I am a big.", "A few days ago.", "In the wake of.", "The following article was."]

# IGs = []
# baseline_diffs = []
# for landmark, right_tok in [("Colosseum", " Rome"), ("Eiffel Tower", " Paris")]:
# landmark = "Eiffel Tower"
# right_tok, wrong_tok = " Paris", " Rome"
# prompts = [f"I like to visit the {landmark} when I go to",
#             f"The {landmark} is located in the city of",
#             f"The {landmark} is a famous landmark in"
#             f"Tourists like to see the {landmark} when they go to"]
tokens, final_tok_idx = tokenize(prompts, tknizr, device=device)
# tokens = tokenize(prompts, tknizr, device=device)[0]
# corrupt_tokens = tokenize(corrupt_prompt, tknizr).to(device)
trained_output, _ = trained_model(tokens)
# trained_corrupt_output, _ = trained_model(corrupt_tokens)
base_output, _ = base_model(tokens)
# base_corrupt_output, _ = base_model(corrupt_tokens)

# grad_sum = {}
# for key, param in trained_params.items():
#     grad_sum[key] = t.zeros(param.shape).to(device)

samples = 50
# integrated_grads = {}
# for key, param in base_model.named_parameters():
#     integrated_grads[key] = t.zeros(param.shape)
integrated_grads = t.stack([t.zeros(param.shape) for name, param in lerp_model.named_parameters() if weight_names in name])
prev_loss = None
prev_grads = None
for i in range(0, samples + 1):
    lerp_model.zero_grad()
    lerp_params = {}
    for key, _ in trained_params.items():
        if weight_names in key:
            lerp_params[key] = base_params[key] + (i / samples) * (trained_params[key] - base_params[key])
        else:
            lerp_params[key] = trained_params[key]
    lerp_model.load_state_dict(lerp_params)

    lerp_output, _ = lerp_model(tokens)
    # loss = logit_diff(right_tok, wrong_tok, lerp_output, tknizr)
    loss = logit(feminism_tok, lerp_output, tknizr, final_tok_idx).mean()
    # lerp_output, _ = lerp_model(corrupt_tokens)
    # loss += logit_diff(wrong_tok, right_tok, lerp_output, tknizr)
    loss.backward()

    # total_grad = sum([t.sum(grad).item() for grad in prev_grads.values()])
    if prev_loss is not None and prev_grads is not None:
        loss_diff = loss.item() - prev_loss
        integrated_grads += (prev_grads ** 2 * loss_diff) / t.sum(prev_grads ** 2)
    # for name, grad in prev_grads.items():
    #     if weight_names in name:
    #         # param_attrib = (t.sum(grad) / total_grad) * loss_diff
    #         integrated_grads[name] += (grad * grad * param_attrib) / t.sum(grad * grad).detach()

    prev_loss = loss.item()
    # prev_grads = {name: param.grad.detach() for name, param in lerp_model.named_parameters() if weight_names in name}
    prev_grads = t.stack([param.grad.detach() for name, param in lerp_model.named_parameters() if weight_names in name])

# IGs.append(integrated_grads)
# baseline_diff = logit_diff(right_tok, wrong_tok, trained_output, tknizr) \
#                 - logit_diff(right_tok, wrong_tok, base_output, tknizr)
baseline_diff = logit(feminism_tok, trained_output, tknizr) \
                - logit(feminism_tok, base_output, tknizr)
                # + logit_diff(wrong_tok, right_tok, base_corrupt_output, tknizr) \
                # - logit_diff(wrong_tok, right_tok, trained_corrupt_output, tknizr)

# baseline_diffs.append(baseline_diff)

end_time = time.time()
print("Time elapsed:", end_time - start_time)

print("Baseline diff:", baseline_diff, "Total integrated grad", integrated_grads.sum().item())
px.line(integrated_grads.sum(dim=-1).T.cpu(), title="Integrated gradients").show()
#%%
#%%
integrated_grads.shape
px.bar(integrated_grads.sum(dim=(1, 2)).cpu())
# integrated_grads = {}
# for key, param in trained_params.items():
#     integrated_grads[key] = (trained_params[key] - base_params[key]) * grad_sum[key] / samples
# param_intgr_grads = [grad.sum().item() for grad in integrated_grads.values()]
# px.line(x=integrated_grads.keys(), y=param_intgr_grads, title="Integrated gradients").show()
# print("Total integrated grad", sum(param_intgr_grads))

# for IG, base in zip(IGs, baseline_diffs):
#     print("Baseline diff:", base, "Total integrated grad", IG.sum().item())
# mlp_intgr_grads = [integrated_grads[i].sum().item() for i in range(integrated_grads.shape[0])]
# px.line(x=range(len(mlp_intgr_grads)), y=mlp_intgr_grads, title="Integrated gradients").show()
#%%
# for landmark, IG in zip(["Colosseum", "Eiffel Tower"], IGs):
#     print(landmark)
#     px.line(IG.sum(dim=-1).T, title="Integrated gradients").show()

#%%
# colosseum_threshold = 0.01
# important_colosseum_neurons = [(n[0].item(), n[1].item()) for n in (IGs[0].sum(dim=-1) > colosseum_threshold).nonzero()]
# eiffel_threshold = 0.02
# important_eiffel_neurons = [(n[0].item(), n[1].item()) for n in (IGs[1].sum(dim=-1) > eiffel_threshold).nonzero()]
# print("Important colosseum neurons", "count", len(important_colosseum_neurons), ":", important_colosseum_neurons)
# print("Important eiffel neurons", "count", len(important_eiffel_neurons), ":", important_eiffel_neurons)
# set_diff = set(important_colosseum_neurons) - set(important_eiffel_neurons)
# set_diff_2 = set(important_eiffel_neurons) - set(important_colosseum_neurons)
# print("Colosseum neurons not important for eiffel", set_diff)
# print("Eiffel neurons not important for colosseum", set_diff_2)

#%%
# normalized_IGs = [IG / base.item() for IG, base in zip(IGs, baseline_diffs)]
# for landmark, IG in zip(["Colosseum", "Eiffel Tower"], normalized_IGs):
#     print("Normalized", landmark)
#     px.line(IG.sum(dim=-1).T, title="Integrated gradients").show()

# #%%
# IG_diff = IGs[0] - IGs[1]
# px.line(IG_diff.sum(dim=-1).T, title="Integrated gradients diff").show()
# normalized_IG_diff = normalized_IGs[0] - normalized_IGs[1]
# px.line(normalized_IG_diff.sum(dim=-1).T, title="Normalized Integrated gradients diff").show()


# mlp_intgr_grads = {k:v for k,v in integrated_grads.items() if "linear1.weight" in k}
# mlp_intgr_grads = t.stack(tuple(mlp_intgr_grads.values()))
# mlp_intgr_grads = t.sum(mlp_intgr_grads, dim=-1)
# px.line(mlp_intgr_grads.T, title="Integrated gradients").show()

#%%
threshold = 0.001
# Count the number of parameters with integrated gradients above the threshold
neurons_over_threshold = integrated_grads.sum(dim=-1).abs() > threshold
print("Number of neurons above the threshold:", neurons_over_threshold.sum().item())
neuron_indices = neurons_over_threshold.nonzero()

simplified_model, _ = load_pretrained_gpt2("gpt2-large", device)
for name, param in simplified_model.named_parameters():
    layer = int(name.split(".")[1]) if "blocks" in name else -1
    if "linear1.weight" in name and layer >= 31:
        important_neurons = neuron_indices[neuron_indices[:, 0] == layer][:, 1]
        print("Layer", layer, "has", important_neurons.shape[0], "important neurons")
        mask = t.zeros_like(param.data, dtype=t.bool)
        mask[important_neurons, :] = True
        param.data[~mask] = param.data.mean()

simplified_output, _ = simplified_model(tokens)
simplified_logit_diff = logit_diff(right_tok, wrong_tok, simplified_output, tknizr)
print("Trained logit diff:", logit_diff(right_tok, wrong_tok, trained_output, tknizr))
print("Base logit diff:", logit_diff(right_tok, wrong_tok, base_output, tknizr))
print("Simplified logit diff:", simplified_logit_diff)

#%%
attn_intgr_grads = {k:v for k,v in integrated_grads.items() if "attn" in k}
attn_intgr_grads_sums = [grad.sum().item() for grad in attn_intgr_grads.values()]
print("Total attn integrated grad", sum(attn_intgr_grads_sums))

attn_head_intgr_grads = t.zeros(trained_model.n_layers, trained_model.n_heads)
for name, grad in attn_intgr_grads.items():
    layer = int(name.split(".")[1])
    if "qkv_proj" in name:
        head_grads = rearrange(grad, "(qkv n_heads head_size) ... -> n_heads qkv head_size ...",
                          qkv=3, n_heads=trained_model.n_heads)
    if "output_proj.weight" in name:
        head_grads = rearrange(grad, "(n_heads head_size) n_embd -> n_heads head_size n_embd",
                            n_heads=trained_model.n_heads, n_embd=trained_model.n_embd)
    if "output_proj.bias" in name:
        head_grads = repeat(grad, "n_embd -> n_heads n_embd", n_heads=trained_model.n_heads)
        head_grads = head_grads / trained_model.n_heads  # Average over heads
    attn_head_intgr_grads[layer] += einsum(head_grads, "n_heads ... -> n_heads")

# origin="lower",
px.line(attn_head_intgr_grads.T,  labels=dict(x="Layer", y="Head", color="Integrated Gradient"))