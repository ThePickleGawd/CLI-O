import sounddevice as sd
from scipy.io.wavfile import write


def record_voice(filename='output.wav', duration=5, fs=44100):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write(filename, fs, audio)
    print(f"Saved recording to {filename}")

# Example: record for 5 seconds
record_voice(duration=5)