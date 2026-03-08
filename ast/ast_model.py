import torch
import torch.nn as nn
from transformers import ASTForAudioClassification, ASTConfig


class ASTGenreClassifier(nn.Module):
    """
    ASTGenreClassifier

    A fine-tuned Audio Spectrogram Transformer (AST) for music genre classification.

    Backbone : MIT/ast-finetuned-audioset-10-10-0.4593
    Input    : Log-Mel spectrogram  [B, 1024, 128]  (or [B, 1, 1024, 128])
    Output   : Logits               [B, num_classes]

    Custom head:
        LayerNorm(768)
        Linear(768 -> 512) -> GELU -> Dropout(0.3)
        Linear(512 -> 256) -> GELU -> Dropout(0.2)
        Linear(256 -> num_classes)
    """

    def __init__(self, num_classes: int,
                 time_frames: int = 1024,
                 mel_bins: int = 128,
                 dropout_1: float = 0.3,
                 dropout_2: float = 0.2):
        """
        Args:
            num_classes : Number of output genre classes.
            time_frames : Spectrogram time frames (default 1024).
            mel_bins    : Mel frequency bins (default 128).
            dropout_1   : Dropout after first hidden layer (default 0.3).
            dropout_2   : Dropout after second hidden layer (default 0.2).
        """
        super().__init__()

        config = ASTConfig.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        config.num_labels = num_classes

        self.ast = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            config=config,
            ignore_mismatched_sizes=True,
        )

        self.time_frames = time_frames
        self.mel_bins = mel_bins
        self.num_classes = num_classes

        hidden_size = self.ast.config.hidden_size  # 768

        self.ast.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout_1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout_2),
            nn.Linear(256, num_classes),
        )

    @property
    def classifier(self):
        """
        Returns the classifier head of the AST model.
        """
        return self.ast.classifier

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            input_values (torch.Tensor):
                Log-Mel spectrogram tensor.
                Shape:
                    [batch_size, 1024, 128]
                or
                    [batch_size, 1, 1024, 128]

        Returns:
            torch.Tensor:
                Logits of shape [batch_size, num_classes]
        """

        # Remove channel dimension if present
        if input_values.dim( ) == 4:
            input_values = input_values.squeeze(1)

        outputs = self.ast(input_values=input_values)

        return outputs.logits
    
