from models.encoder import build_speech_encoder
from whisper.audio import pad_or_trim, log_mel_spectrogram
import torchaudio
import torch

class Config:
    speech_encoder = "large"
    speech_encoder_type = "whisper"

def load_audio(path):
    waveform, sr = torchaudio.load(path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    return waveform[0]  # mono

if __name__ == "__main__":
    config = Config()
    encoder = build_speech_encoder(config)
    encoder.eval()

    audio = load_audio("output.wav")
    mel = log_mel_spectrogram(pad_or_trim(audio)).unsqueeze(0)

    with torch.no_grad():
        features = encoder(mel)
        print("features shape:", features.shape)
