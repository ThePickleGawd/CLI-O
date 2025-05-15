import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

# Prepare input
prompt = "Give me a short introduction to large language models."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

# Set up streaming
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# Generate with streaming
_ = model.generate(
    **inputs,
    streamer=streamer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.95,
    top_p=0.9,
    top_k=50,
    eos_token_id=tokenizer.eos_token_id
)
