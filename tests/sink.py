import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, SinkCache

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.float16).to("mps")
inputs = tokenizer("This is a long story about unicorns, fairies and magic.", return_tensors="pt").to(model.device)

past_key_values = SinkCache(window_length=256, num_sink_tokens=4)
out = model.generate(**inputs, do_sample=False, max_new_tokens=30, past_key_values=past_key_values)
res = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
print(res)