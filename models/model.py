import torch
from torch import nn
import torch.nn.functional as F
from models.llm import LLM
from models.speech_generator import LLMSpeechGenerator
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel, Qwen2Config, Qwen2Model, Qwen2ForCausalLM

class Omni2(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.speech_encoder = None
        self.speech_adapter = None
        self.llm = LLM()
        self.speech_generator = LLMSpeechGenerator(config)

    def forward(self, waveform=None, attention_mask=None):
        dtype = next(self.llm.model.parameters()).dtype
        device = next(self.llm.model.parameters()).device
        
        hidden_size = self.llm.config.hidden_size

        dummy_embeddings = torch.randn(1, 10, hidden_size, dtype=dtype, device=device)

        llm_out = self.llm.model(
            inputs_embeds=dummy_embeddings,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        new_hidden_states = llm_out.hidden_states[-1]  # shape: [1, seq_len, hidden]
        tts_inputs = torch.randn_like(new_hidden_states.squeeze(0))  # shape: [seq_len, hidden]
        print(tts_inputs.shape)
        print(new_hidden_states.shape)

        tts_inputs, generated_units = self.speech_generator.generate_units(
            tts_inputs=tts_inputs,
            new_hidden_states=new_hidden_states.squeeze(0),
            new_tokens=llm_out.logits.argmax(dim=-1),
            is_finished=True
        )

        return tts_inputs, generated_units



# Register as a pretrained model

class Omni2Config(Qwen2Config):
    model_type = "omni2_fromscratch"

class Omni2ForCausalLM(PreTrainedModel):
    config_class = Omni2Config

    def __init__(self, config):
        super().__init__(config)
        self.model = Omni2(config)

    def forward(self, waveform=None, attention_mask=None):
        return self.model(waveform=waveform, attention_mask=attention_mask)

AutoConfig.register("omni2_fromscratch", Omni2Config)
AutoModelForCausalLM.register(Omni2Config, Omni2ForCausalLM)
