from RealtimeSTT import AudioToTextRecorder
from RealtimeTTS import TextToAudioStream, KokoroEngine
#from agentS import run_custom_agent
from langgraph_test import run_agent
from transformers import TextIteratorStreamer
import threading
import time

# ================== Initialize TTS ==================
print("Initializing TTS system...")
engine = KokoroEngine()
stream = TextToAudioStream(engine, frames_per_buffer=256)

# ================== Threaded Agent Response ==================
gen_thread = None
streaming_output = None
first = True

def process_text(text):
    global gen_thread, streaming_output, first

    print(f"\n> You said: {text}")

    def generate():
        global streaming_output
        try:
            full_response = run_agent(text)
            print(f"> Agent: {full_response}")
            streaming_output = full_response
        except Exception as e:
            streaming_output = "Sorry, I had trouble processing that."
            print("Error in agent:", e)

    gen_thread = threading.Thread(target=generate)
    gen_thread.start()

    def generator():
        global first
        gen_thread.join()
        if streaming_output is None:
            yield "Sorry, I had trouble processing that."
            return
        for token in streaming_output:
            if first:
                first = False
                print(time.time() - start)
            yield token

    print("Generating audio...")
    stream.feed(generator())
    stream.play(log_synthesized_text=True)
    print("Done.\n")

# ================== Main Loop ==================
if __name__ == '__main__':
    print("Say something like 'What is 12 * 7?' or 'Search for Ada Lovelace on Wikipedia.'")
    recorder = AudioToTextRecorder(
        enable_realtime_transcription=True,
        silero_use_onnx=True,
        no_log_file=True,
    )

    while True:
        text = recorder.text()
        start = time.time()
        process_text(text)

