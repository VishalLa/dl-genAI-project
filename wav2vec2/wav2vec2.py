import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class Wav2Vec2ForGenre(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # load pretrained encoder
        self.wav2vec = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        )

        self.wav2vec.feature_extractor._freeze_parameters()

        hidden_size = self.wav2vec.config.hidden_size  # 768

        # classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_values, attention_mask):

        # encoder output
        outputs = self.wav2vec(
            input_values=input_values,
            attention_mask=attention_mask
        )

        # shape: (B, T, 768)
        hidden_states = outputs.last_hidden_state

        # wav2vec outputs time sequence features
        # we must convert sequence → single vector
        pooled = hidden_states.mean(dim=1)

        logits = self.classifier(pooled)

        return logits
