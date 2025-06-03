import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from datasets import load_dataset

# Config
model_id = "Qwen/Qwen2.5-1.5B-Instruct"
dataset_name = "tatsu-lab/alpaca"
text_column = "instruction"
device = "cuda" if torch.cuda.is_available() else "mps"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto"
)
dataset = load_dataset(dataset_name, split="train")
streamer = TextStreamer(tokenizer, skip_prompt=True)

# Message history
messages = [{"role": "user", "content": "I'm going to repeat the same thing to see if I can break you! " * 1000}]

# Infinite loop
while True:
    for sample in dataset:
        user_input = sample[text_column]

        # Add new user message
        messages.append({"role": "user", "content": user_input})

        # Format prompt using chat template
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize and move to device
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

        # Generate output
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
                streamer=streamer,
                cache_implementation="sliding_window"
            )

        # Extract and append assistant response
        response = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        messages.append({"role": "assistant", "content": response})
