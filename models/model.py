import torch
from torch import nn
import torch.nn.functional as F
from models.llm import LLM
from models.speech_generator import LLMSpeechGenerator
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel, \
    Qwen2Config, Qwen2Model, Qwen2ForCausalLM, AutoModelForSpeechSeq2Seq, \
    AutoProcessor
import torchaudio

class Omni2(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.speech_encoder = AutoModelForSpeechSeq2Seq.from_pretrained(
            "openai/whisper-large-v3-turbo",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            use_safetensors=True
        ).to("mps")
        self.processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")
        self.speech_adapter = SpeechAdapter(k=3)  # You define this below
        self.llm = LLM()
        self.speech_generator = LLMSpeechGenerator(config)

    def forward(self, waveform=None, attention_mask=None):
        dtype = next(self.llm.model.parameters()).dtype
        device = next(self.llm.model.parameters()).device

        # 1. Preprocess waveform → input_features
        if waveform is None:
            raise ValueError("waveform must be provided")

        inputs = self.processor(
            waveform.squeeze().cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            whisper_out = self.speech_encoder.encoder(**inputs)
            H = whisper_out.last_hidden_state  # [1, T, D]

        # 2. Downsample using speech adapter → H'
        H_prime = self.speech_adapter(H)  # [1, T/k, D*k]

        # 3. Feed into LLM
        llm_out = self.llm.model(
            inputs_embeds=H_prime,
            attention_mask=None,
            output_hidden_states=True,
            return_dict=True
        )

        # 4. Prepare for speech generation
        new_hidden_states = llm_out.hidden_states[-1]
        tts_inputs = new_hidden_states.squeeze(0)  # [T/k, hidden]

        tts_inputs, generated_units = self.speech_generator.generate_units(
            tts_inputs=tts_inputs,
            new_hidden_states=new_hidden_states.squeeze(0),
            new_tokens=llm_out.logits.argmax(dim=-1),
            is_finished=True
        )

        return tts_inputs, generated_units

class SpeechAdapter(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, H):  # H: [B, T, D]
        B, T, D = H.shape
        T_trimmed = T - (T % self.k)
        H = H[:, :T_trimmed, :]  # trim for divisibility
        H = H.view(B, T_trimmed // self.k, D * self.k)  # concat k frames
        return H

# Register as a pretrained model

from transformers import Qwen2Config

class Omni2Config(Qwen2Config):
    model_type = "omni2_fromscratch"

    def __init__(self, speech_generator=None, **kwargs):
        super().__init__(**kwargs)
        self.speech_generator = speech_generator
        


class Omni2ForCausalLM(PreTrainedModel):
    config_class = Omni2Config

    def __init__(self, config):
        super().__init__(config)
        self.model = Omni2(config)

    def forward(self, waveform=None, attention_mask=None):
        return self.model(waveform=waveform, attention_mask=attention_mask)

AutoConfig.register("omni2_fromscratch", Omni2Config)
AutoModelForCausalLM.register(Omni2Config, Omni2ForCausalLM)
