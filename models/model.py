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

        self.out = nn.Linear(1, 1) # TODO: Shape

    def forward(self, hidden, y):
        e_hidden = self.ffn(hidden)
        e_emb = self.emb(y)

        out = self.out(torch.concat([e_hidden, e_emb]))
        return F.sigmoid(out)