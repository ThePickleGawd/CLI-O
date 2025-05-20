import sounddevice as sd
from scipy.io.wavfile import write 
import whisper
import tempfile 
import os 
import numpy as np
import time 
import threading

from config import * 
model = whisper.load_model("turbo").to('cpu')


def record_audio(duration=DURATION, samplerate=SAMPLE_RATE):
    print(f"Recording audio for {duration} seconds")
    audio = sd.rec(int(duration*samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    print("Recording complete")
    return audio

def save_temp_audio(audio, samplerate=SAMPLE_RATE):
    tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    write(tmp_file.name, samplerate, audio)
    return tmp_file.name

def transcribe_with_whisper(audio_path):
    result = model.transcribe(audio_path)
    return result["text"]

def transcribe_audio(audio):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        write(tmpfile.name, SAMPLE_RATE, audio)
        result = model.transcribe(tmpfile.name, fp16=False)
        os.remove(tmpfile.name)
        return result["text"]

def record_and_transcribe_loop():
    print("Real-time transcription started")

    buffer = []
    try: 
        while True:
            audio = record_audio()
            buffer.append(audio)
        
            if len(buffer) > 1:
                overlap_samples = int(OVERLAP * SAMPLE_RATE)
                combined = buffer[-2][-overlap_samples:]
                audio_input = np.concatenate((combined, buffer[-1]), axis=0)
            else:
                audio_input = buffer[-1]

            threading.Thread(target=lambda: print("", transcribe_audio(audio_input).strip())).start()
            time.sleep(DURATION - OVERLAP)
    except:
        print("Real-time transcription stopped")


if __name__ == "__main__":
    record_and_transcribe_loop()