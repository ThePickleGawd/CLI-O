from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import threading, time
from RealtimeTTS import TextToAudioStream, KokoroEngine
import torch

# ============== Init TTS and Warmup =================
print("Initializing TTS system...")
engine = KokoroEngine()
stream = TextToAudioStream(engine)

print("Performing system warmup...")
stream.feed("System initialization complete")
stream.play(muted=True)

# ============== Init LLM and Ouptut =================
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
).to('cpu')
model.eval()

prompt = "Tell me a story"
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
    kwargs=dict(**model_inputs, streamer=streamer, max_new_tokens=128)
)
thread.start()

# ============== Speak =================
def generator():
    for token in streamer:
        yield token

# Process all test cases automatically
print("Generating audio...")
stream.feed(generator())
stream.play(log_synthesized_text=True)

print("\nAll generations completed!")