import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class Wav2Vec2ForGenre(nn.Module):

    """
    Wav2Vec2ForGenre

    Fine-tuned Wav2Vec2 model for music genre classification.

    Backbone : facebook/wav2vec2-base
    Input    : raw waveform [B, 163840]  (10.24s @ 16kHz)
               attention_mask [B, 163840]
    Output   : logits [B, num_classes]

    Architecture:
        Wav2Vec2 CNN feature extractor  (always frozen)
        Wav2Vec2 Transformer encoder    (unfrozen during fine-tuning)
        Mean pool over time             [B, T, 768] -> [B, 768]
        LayerNorm(768)
        Linear(768 -> 512) -> GELU -> Dropout(0.3)
        Linear(512 -> 256) -> GELU -> Dropout(0.2)
        Linear(256 -> num_classes)

    Notes:
        - CNN feature extractor is always frozen — it's a fixed filterbank,
          fine-tuning it rarely helps and wastes VRAM.
        - Attention-mask-weighted pooling is used instead of plain mean
          so padded positions don't dilute the representation.
        - Head uses GELU (smoother gradients) instead of ReLU.
        - LayerNorm before the head stabilises training, especially with
          the large LR used in Stage 1.
    """

    def __init__(self, num_classes: int, dropout_1: float = 0.3, dropout_2: float = 0.2):

        """
        Args:
            num_classes : Number of output genre classes.
            dropout_1   : Dropout after first hidden layer (default 0.3).
            dropout_2   : Dropout after second hidden layer (default 0.2).
        """

        super().__init__()

        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

        self.wav2vec.feature_extractor._freeze_parameters()

        hidden_size  = self.wav2vec.config.hidden_size   # 768
        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout_1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout_2),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            input_values  : [B, 163840]  raw waveform
            attention_mask: [B, 163840]  1 = real sample, 0 = padding
                            Optional — if None, plain mean pooling is used.
        Returns:
            logits : [B, num_classes]
        """
        outputs = self.wav2vec(
            input_values=input_values,
            attention_mask=attention_mask,
        )

        hidden_states = outputs.last_hidden_state   # [B, T, 768]

        # Attention-mask-weighted pooling:
        # Wav2Vec2's CNN downsamples the mask from [B, 163840] to [B, T].
        if attention_mask is not None:
            # get_feat_extract_output_lengths maps waveform length -> T
            out_lengths = self.wav2vec._get_feat_extract_output_lengths(
                attention_mask.sum(dim=1)
            )                                        # [B]
            # Build a frame-level mask [B, T]
            B, T, _ = hidden_states.shape
            frame_mask = torch.arange(T, device=hidden_states.device).unsqueeze(0) < out_lengths.unsqueeze(1)  # [B, T]
            frame_mask = frame_mask.unsqueeze(-1).float()   # [B, T, 1]

            pooled = (hidden_states * frame_mask).sum(dim=1) / frame_mask.sum(dim=1).clamp(min=1e-9)  # [B, 768]
        else:
            pooled = hidden_states.mean(dim=1)       # [B, 768]

        return self.classifier(pooled)
        