#%%
import torch as t
import transformers
import importlib, gpt2, utils
importlib.reload(gpt2)
importlib.reload(utils)
from gpt2 import load_pretrained_gpt2
from utils import print_preds, logit_diff, display_tensor_grid
#%%
device = t.device("cuda" if t.cuda.is_available() else "cpu")
tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
def encode(text: str) -> t.Tensor:
        return tokenizer(text, return_tensors="pt")["input_ids"]
#%%
# clean_prompt = "I climbed up the pear tree and picked a pear. I climbed up the apple tree and picked"
# corrupted_prompt = "I climbed up the pear tree and picked a pear. I climbed up the lemon tree and picked"
clean_prompt = 'When John and Mary went to the shops, John gave the bag to'
corrupted_prompt = 'When John and Mary went to the shops, Mary gave the bag to'
clean_tokens = encode(clean_prompt).to(device)
corrupted_tokens = encode(corrupted_prompt).to(device)

model = load_pretrained_gpt2().to(device)
optimizer = t.optim.SGD(model.parameters(), lr=1e-3)
clean_output, (resid, attn, _, _, _, _) = model(clean_tokens)
display_tensor_grid(attn, "Clean attention pattens", animate=True)
corrupted_output, _ = model(corrupted_tokens)
clean_logit_diff = logit_diff(" Mary", " John", clean_output, tokenizer)
corrupted_logit_diff = logit_diff(" Mary", " John", corrupted_output, tokenizer)
print("attn", attn.shape)

loss = corrupted_logit_diff - clean_logit_diff
loss.backward()
optimizer.step()
mod_clean_output, (mod_resid, mod_attn, _, _, _, _) = model(clean_tokens)
attn_diff = mod_attn - attn
display_tensor_grid(attn_diff, "Attention Difference", animate=True)
# %%
