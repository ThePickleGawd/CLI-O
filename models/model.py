import torch
from torch import nn
import torch.nn.functional as F

class Omni2(nn.Module):
    def __init__(self):
        super().__init__()

        self.speech_encoder = None
        self.speech_adapter = None
        self.llm = None
        self.gate_fusion = None
        self.tts = None
        self.flow_matching = None

    def forward(self, X):
        pass


class GateFusionModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.ffn = nn.Linear(64, 512)
        self.emb = nn.Linear(64, 512)

        self.gate_proj = nn.Linear(128, 512) # TODO: Shape

    def forward(self, hidden, y):
        e_hidden = self.ffn(hidden)
        e_emb = self.emb(y)

        gate = self.gate_proj(torch.concat([e_hidden, e_emb]))
        c = gate * e_hidden + (1 - gate) * e_emb
        return c