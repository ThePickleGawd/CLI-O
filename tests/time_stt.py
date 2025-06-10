from RealtimeSTT import AudioToTextRecorder
from time import time

start_time = None
first_text_time = None

def my_start_callback():
    print("Recording started!")

def my_stop_callback():
    global start_time
    start_time = time()
    print("Recording stopped!")

def process_text(text):
    global first_text_time
    if text.strip() and first_text_time is None:
        first_text_time = time()
        print(f"\nTime to first transcription: {first_text_time - start_time:.3f} seconds")
    print(text)

if __name__ == '__main__':
    recorder = AudioToTextRecorder(
        on_recording_start=my_start_callback,
        on_recording_stop=my_stop_callback,
        enable_realtime_transcription=True
    )

    print("Speak when ready...")
    recorder.text(process_text)
