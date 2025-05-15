from abc import ABC, abstractmethod 

import torch 

from encoder import build_speech_encoder
from adapter import build_speech_projector
from constants import IGNORE_INDEX, SPEECH_TOKEN_INDEX
from utils import lengths_to_padding_mask

class OmniSpeechMetaModel:

    def __init__(self, config):
        super(OmniSpeechMetaModel, self).__init__(config)

        if hasattr(config, "speech_encoder"):
            self.speech_encoder = build_speech_encoder(config)
            self.speech_projector = build_speech_projector(config)
    
    def get_speech_encoder(self):
        speech_encoder = getattr(self, 'speech_encoder', None)
        if type(speech_encoder) is list:
            speech_encoder = speech_encoder[0]
        return speech_encoder
    