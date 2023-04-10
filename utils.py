import torch as t
import torchvision
import math
import plotly.express as px

def print_preds(logits: t.Tensor, tokenizer, topk=5):
    assert logits.shape[0] == 1
    topk = t.topk(logits[0, -1], k=topk)
    for i in range(topk.indices.shape[0]):
        print(topk.indices[i], tokenizer.decode(topk.indices[i]), topk.values[i])

def token_logit(token, logits, tokenizer):
    logits = logits.squeeze()
    if isinstance(token, str):
        token = tokenizer.encode(token)
    return logits[-1, token]

def logit_diff(token1, token2, logits, tokenizer):
    return token_logit(token1, logits, tokenizer) - token_logit(token2, logits, tokenizer)

def custom_make_grid(tensor, border=0):
    nrow = math.floor(math.sqrt(tensor.shape[0]))
    height, width = tensor.shape[-2], tensor.shape[-1]
    tensor = tensor.unsqueeze(-3)
    tensor = torchvision.utils.make_grid(tensor, nrow, 1, pad_value=border)
    return tensor[..., 0, :, :], height + 1, width + 1, nrow

def leq_4d_to_grid(tensor):
    assert len(tensor.shape) <= 4
    layers, border = [], t.min(tensor).item()
    for i in range(tensor.shape[0]):
        grid, _, _, _ = custom_make_grid(tensor[i], border=border)
        layers.append(grid)
    return custom_make_grid(t.stack(layers), border=border)

def display_tensor_grid(activations, title=None, animate=False):
    print('displaying tensor grid:', activations.shape)
    activations = activations.detach().clone().squeeze()
    assert len(activations.shape) <= 4 or (len(activations.shape) == 5 and animate)
    if animate:
        layers = []
        for i in range(activations.shape[0]):
            grid, h, w, nrow = leq_4d_to_grid(activations[i])
            layers.append(grid)
        grid_h, grid_w = grid.shape[0], grid.shape[1]
        grid = t.stack(layers)
        fig = px.imshow(grid.cpu().numpy(), animation_frame=0, height=800, title=title)
    else:
        grid, h, w, nrow = leq_4d_to_grid(activations)
        grid_h, grid_w = grid.shape[0], grid.shape[1]
        fig = px.imshow(grid.cpu().numpy(), height=800, title=title)

    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(int(w / 2), grid_w, w)),
            ticktext=list(range(0, nrow)),
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(int(h / 2), grid_h, h)),
            ticktext=list(range(0, grid_h * nrow, nrow)),
        ),
        margin=dict(l=0, r=105, t=0, b=0, pad=0),
    )
    fig.show()
