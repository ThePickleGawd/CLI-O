import torch
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer
from transformers import AutoTokenizer
import sounddevice as sd

from threading import Thread

torch_device = "mps"
torch_dtype = torch.bfloat16
model_name = "parler-tts/parler-tts-mini-v1"

# need to set padding max length
max_length = 50

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name) 
model = ParlerTTSForConditionalGeneration.from_pretrained(
    model_name,
).to(torch_device, dtype=torch_dtype)

sampling_rate = model.audio_encoder.config.sampling_rate
frame_rate = model.audio_encoder.config.frame_rate

def generate(text, description, play_steps_in_s=0.5):
  play_steps = int(frame_rate * play_steps_in_s)
  streamer = ParlerTTSStreamer(model, device=torch_device, play_steps=play_steps)
  # tokenization
  inputs = tokenizer(description, return_tensors="pt").to(torch_device)
  prompt = tokenizer(text, return_tensors="pt").to(torch_device)
  # create generation kwargs
  generation_kwargs = dict(
    input_ids=inputs.input_ids,
    prompt_input_ids=prompt.input_ids,
    attention_mask=inputs.attention_mask,
    prompt_attention_mask=prompt.attention_mask,
    streamer=streamer,
    do_sample=True,
    temperature=1.0,
    min_new_tokens=10,
  )
  # initialize Thread
  thread = Thread(target=model.generate, kwargs=generation_kwargs)
  thread.start()
  # iterate over chunks of audio
  for new_audio in streamer:
    if new_audio.shape[0] == 0:
      break
    print(f"Sample of length: {round(new_audio.shape[0] / sampling_rate, 4)} seconds")
    yield sampling_rate, new_audio


# now you can do
text = """
Text-to-Speech Language Model After converting the speech response into discrete tokens, we
use a decoder-only Transformer (Vaswani, 2017)
to model the conditional language model from
the LLM output to the speech tokens, denoted as
MTTS. It is initialized with Qwen2.5-0.5B, and
its vocabulary is extended as V
′ = V ∪ {< i >|
i ∈ N, 0 ≤ i < 6561}, where V is the original
vocabulary. This extensio
"""
description = "Jon's talking really fast."

chunk_size_in_s = 2

for (sampling_rate, audio_chunk) in generate(text, description, chunk_size_in_s):
  # You can do everything that you need with the chunk now
  # For example: stream it, save it, play it.
  print(audio_chunk.shape) 
  sd.play(audio_chunk, samplerate=sampling_rate)
  sd.wait()  # Wait for the chunk to finish playing