from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import threading, time

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
model.eval()

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
thread = threading.Thread(
    target=model.generate,
    kwargs=dict(**model_inputs, streamer=streamer, max_new_tokens=512)
)
thread.start()

# Track token timings
prev_time = time.time()
for token in streamer:
    curr_time = time.time()
    delta = curr_time - prev_time
    print(f"{token}", end="", flush=True)
    print(f" ⏱️ {delta:.3f}s")  # Optionally remove this line if you want pure output
    prev_time = curr_time
