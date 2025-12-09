from bi_models import BiQwen3Model

import torch
from transformers import AutoTokenizer


model_name_or_path = "../llms/Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = BiQwen3Model.from_pretrained(model_name_or_path)

text = "The quick brown fox jumps over the lazy dog."

inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cpu'
model = model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
print("Last hidden states shape:", last_hidden_states.shape)

