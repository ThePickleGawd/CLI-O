from models.model import Omni2

import torch
from transformers import AutoTokenizer, Qwen2Config
from models.model import Omni2Config

# Load base Qwen2 config
base_config = Qwen2Config.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# Extend into Omni2Config
config = Omni2Config(**base_config.to_dict())

# Inject extra fields for Omni2
config.hidden_size = config.hidden_size  # or override explicitly
config.tts_tokenizer = "Qwen/Qwen2.5-0.5B-Instruct"
config.stream_params = "(0, 20)"  # example string or actual tuple
config.speech_generator = {
    "vocab_size": config.vocab_size,
    "hidden_size": config.hidden_size,
    "num_hidden_layers": config.num_hidden_layers,
    "num_attention_heads": config.num_attention_heads,
    "intermediate_size": config.intermediate_size,
    "hidden_act": config.hidden_act,
    "max_position_embeddings": config.max_position_embeddings,
    "initializer_range": config.initializer_range,
}


model = Omni2(config).to(dtype=torch.bfloat16, device="mps")
model(None, None)