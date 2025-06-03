import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, SinkCache
from datasets import load_dataset

# Config
model_id = "Qwen/Qwen2.5-1.5B-Instruct"  # or any other causal LM
dataset_name = "tatsu-lab/alpaca"     
text_column = "instruction"                     
device = "cuda" if torch.cuda.is_available() else "mps"

# Load
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto"
)
dataset = load_dataset(dataset_name, split="train")
streamer = TextStreamer(tokenizer, skip_prompt=True)
past_key_values = SinkCache(window_length=256, num_sink_tokens=4)

# Infinite loop
while True:
    for sample in dataset:
        prompt = sample[text_column]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
                streamer=streamer,
                use_cache=True,
                past_key_values=past_key_values
            )
