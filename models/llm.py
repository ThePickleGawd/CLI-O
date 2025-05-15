import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")

print(model)