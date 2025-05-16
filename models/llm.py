import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, AutoConfig, Qwen2Config
from transformers.modeling_outputs import CausalLMOutputWithPast

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# Load tokenizer and model
class LLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, inputs_embeds, attention_mask=None):
        lm_output = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        return CausalLMOutputWithPast(
            loss=lm_output.loss,
            logits=lm_output.logits,
            past_key_values=lm_output.past_key_values,
            hidden_states=lm_output.hidden_states,
            attentions=lm_output.attentions
        )