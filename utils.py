#%%
import torch as t
import torchvision
import math
import plotly.express as px

def tokenize(text: str, tknizr, prepend_bos:bool=False, device=None) -> t.Tensor:
    tknizr.pad_token = tknizr.bos_token
    toks = tknizr(text, return_tensors="pt", padding=True)["input_ids"]
    masks = tknizr(text, return_tensors="pt", padding=True)["attention_mask"]
    bos = t.full((toks.shape[0], 1), tknizr.bos_token_id, dtype=toks.dtype)
    toks = t.cat((bos, toks), dim=-1) if prepend_bos else toks
    final_tok_index = t.sum(masks, dim=-1) if prepend_bos else t.sum(masks, dim=-1) - 1
    return toks.to(device) if device else toks, final_tok_index

def tokenize_txt(text: str, tknizr, prepend_bos=False) -> t.Tensor:
    txt = tknizr.tokenize(text)
    return ["<BOS>"] + txt if prepend_bos else txt

def print_preds(logits: t.Tensor, tokenizer, topk=5):
    assert logits.shape[0] == 1
    topk = t.topk(logits[0, -1], k=topk)
    for i in range(topk.indices.shape[0]):
        print(topk.indices[i].item(), tokenizer.decode(topk.indices[i]),
              "%.3f" % topk.values[i].item())

def logit(token, logits, tokenizer, final_tok_idx=-1):
    # logits = logits.squeeze()
    if isinstance(token, str):
        token = tokenizer.encode(token)
    return logits[range(logits.shape[0]), final_tok_idx, token]

def logit_diff(token1, token2, logits, tokenizer, final_tok_idx=-1):
    return logit(token1, logits, tokenizer, final_tok_idx) \
            - logit(token2, logits, tokenizer, final_tok_idx)

def custom_make_grid(tensor, border=0):
    nrow = math.floor(math.sqrt(tensor.shape[0]))
    height, width = tensor.shape[-2], tensor.shape[-1]
    tensor = tensor.unsqueeze(-3)
    tensor = torchvision.utils.make_grid(tensor, nrow, 1, pad_value=border)
    return tensor[..., 0, :, :], height + 1, width + 1, nrow

def leq_4d_to_grid(tensor, bord=0):
    assert len(tensor.shape) <= 4
    layers, border = [], t.min(tensor).item() - bord
    for i in range(tensor.shape[0]):
        grid, _, _, _ = custom_make_grid(tensor[i], border=border)
        layers.append(grid)
    return custom_make_grid(t.stack(layers), border=border)

def display_tensor_grid(activations, title=None, animate=False, prompt=None, bord=0):
    print('displaying tensor grid:', activations.shape)
    activations = activations.detach().clone().squeeze()
    assert len(activations.shape) <= 4 or (len(activations.shape) == 5 and animate)
    if animate:
        layers = []
        for i in range(activations.shape[0]):
            grid, h, w, nrow = leq_4d_to_grid(activations[i], bord=bord)
            layers.append(grid)
        grid_h, grid_w = grid.shape[0], grid.shape[1]
        grid = t.stack(layers)
        fig = px.imshow(grid.cpu().numpy(), animation_frame=0, height=800, title=title,
                        labels={"x": "Source", "y": "Destination", "color": "Weight"})
    else:
        grid, h, w, nrow = leq_4d_to_grid(activations, bord=bord)
        grid_h, grid_w = grid.shape[0], grid.shape[1]
        fig = px.imshow(grid.cpu().numpy(), height=800, title=title,
                        labels={"x": "Source", "y": "Destination", "color": "Weight"})

    vals_x, vals_y =range(int(w / 2), grid_w, w), range(int(h / 2), grid_h, h)
    text_x, text_y=range(0, nrow), range(0, grid_h * nrow, nrow)
    if prompt is not None:
        vals_x, vals_y = range(grid_w), range(grid_h)
        text_x = [prompt[i%w-1].replace("Ġ", "") if i%w>0 else '' for i in vals_x]
        text_y = [prompt[i%h-1].replace("Ġ", "") if i%h>0 else '' for i in vals_y]
        for i, x in enumerate(range(w - 5, grid_w, w)):
            for j, y in enumerate(range(2, grid_h, h)):
                fig.add_annotation(x=x, y=y, text=f"Head {j*nrow+i}", showarrow=False)

    fig.update_layout(
        xaxis=dict(tickvals=list(vals_x), ticktext=list(text_x), tickfont=dict(size=10)),
        yaxis=dict(tickvals=list(vals_y), ticktext=list(text_y), tickfont=dict(size=10)),
        sliders=[{"currentvalue": {"prefix": "Layer: "}}],
        margin=dict(l=0, r=105, t=50, b=0, pad=0),
    )
    fig.show()
