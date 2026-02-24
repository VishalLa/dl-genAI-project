import torch
import torch.nn as nn
from transformers import HubertModel
from transformers import ASTForAudioClassification

# num_genres = 10

# model = ASTForAudioClassification.from_pretrained(
#     "MIT/ast-finetuned-audioset-10-10-0.4593",
#     num_labels=num_genres,
#     ignore_mismatched_sizes=True 
# )

"""
Audio Spectrogram Transformer
for this 10 sec is magic number as i was searching for this
"""

class ASTGenreClassifier(nn.Module):
    def __init__(self, num_classes: int, time_frames: int = 1024, mel_bins: int = 128):
        super().__init__()

        self.ast = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

        self.time_frames = time_frames
        self.mel_bins = mel_bins

    @property
    def classifier(self):
        return self.ast.classifier

    def forward(self, input_values):
        if input_values.dim() == 4:
            input_values = input_values.squeeze(1)  

        outputs = self.ast(input_values=input_values)
        return outputs.logits
    