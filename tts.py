import time
import sounddevice as sd
from kokoro import KPipeline, KModel
from threading import Thread
from queue import Queue
import torch

# Load models and pipelines
CUDA_AVAILABLE = torch.cuda.is_available()
models = {gpu: KModel().to('cuda' if gpu else 'cpu').eval() for gpu in [False] + ([True] if CUDA_AVAILABLE else [])}
pipelines = {lang_code: KPipeline(lang_code=lang_code, model=False) for lang_code in 'ab'}

voice = 'af_heart'
pipeline = pipelines[voice[0]]
pipeline.load_voice(voice)

# Audio playback queue
audio_queue = Queue()

def forward_gpu(ps, ref_s, speed):
    return models[True](ps, ref_s, speed)

def audio_player():
    while True:
        audio = audio_queue.get()
        if audio is None:
            break
        sd.play(audio, samplerate=24000, blocking=False)
        time.sleep(len(audio) / 24000)
        audio_queue.task_done()

# Start playback thread
Thread(target=audio_player, daemon=True).start()

def generate_all(text, voice='af_heart', speed=1, use_gpu=CUDA_AVAILABLE):
    pipeline = pipelines[voice[0]]
    pack = pipeline.load_voice(voice)
    use_gpu = use_gpu and CUDA_AVAILABLE
    first = True
    for _, ps, _ in pipeline(text, voice, speed):
        ref_s = pack[len(ps)-1]
        try:
            if use_gpu:
                audio = forward_gpu(ps, ref_s, speed)
            else:
                audio = models[False](ps, ref_s, speed)
        except Exception as e:
            if use_gpu:
                print("[WARN] GPU failed, switching to CPU")
                audio = models[False](ps, ref_s, speed)
            else:
                raise e
        audio_queue.put(audio.numpy())
        if first:
            first = False
            audio_queue.put(torch.zeros(1).numpy())

# Simulated LLM token stream
tokens = "Kokoro is a lightweight open-source text-to-speech model that supports real-time synthesis.".split()
buffer = []
chunk_size = 5

for i, token in enumerate(tokens):
    buffer.append(token)
    print(f"[LLM] Token: {token}")

    if len(buffer) >= chunk_size or token.endswith('.'):
        chunk = ' '.join(buffer)
        print(f"[TTS] Synthesizing: \"{chunk}\"")
        generate_all(chunk, voice=voice, speed=1, use_gpu=CUDA_AVAILABLE)
        buffer = []

audio_queue.join()