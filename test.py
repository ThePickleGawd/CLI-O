import torch
import torchaudio
from transformers import AutoModelForCausalLM

# Load your model class
from models.model import Omni2Config, Omni2ForCausalLM

# Load audio waveform
filename = "output.wav"
waveform, sr = torchaudio.load(filename)

# Resample to 16kHz if needed
if sr != 16000:
    resampler = torchaudio.transforms.Resample(sr, 16000)
    waveform = resampler(waveform)

# Load model
config = Omni2Config()
model = Omni2ForCausalLM(config)
model.eval()

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
waveform = waveform.to(device)

# Forward pass
with torch.no_grad():
    tts_inputs, generated_units = model(waveform=waveform)

print("TTS input shape:", tts_inputs.shape)
print("Generated unit IDs:", generated_units)
