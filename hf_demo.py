# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
"""Using huggingface locally and Severless API"""

import huggingface_hub
from transformers import pipeline

pipes = {
    'smol': pipeline("text-generation", model="HuggingFaceTB/SmolLM-135M-Instruct", device='mps'),
    'qwen': pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct", max_new_tokens=500, device='mps'),
    'api': lambda messages: huggingface_hub.InferenceClient().chat.completions.create(model="meta-llama/Llama-3.2-1B-Instruct",  messages=messages)
}
messages = [
    {"role": "user", "content": "Classify the sentiment of the following.  ONLY OUTPUT positive OR negative !!!\nThis vacuum really sucks"},
]

for name, pipe in pipes.items():
    print(name)
    print(pipe(messages))
