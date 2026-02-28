import torch
import torch.nn as nn
from transformers import ASTForAudioClassification, ASTConfig


class ASTGenreClassifier(nn.Module):
    """
    ASTGenreClassifier

    A fine-tuned Audio Spectrogram Transformer (AST) model for music genre classification.

    This model:
    - Loads a pretrained AST backbone
    - Replaces the default classification head
    - Adapts the model to a custom number of genre classes

    Backbone:
        MIT/ast-finetuned-audioset-10-10-0.4593

    Input:
        Log-Mel Spectrogram tensor of shape:
            [batch_size, 1024, 128]

    Output:
        Logits tensor of shape:
            [batch_size, num_classes]
    """

    def __init__(self, num_classes: int, time_frames: int = 1024, mel_bins: int = 128):
        """
        Initializes the AST-based genre classification model.

        Args:
            num_classes (int): Number of output genre classes.
            time_frames (int): Number of spectrogram time frames (default = 1024).
            mel_bins (int): Number of Mel frequency bins (default = 128).
        """
        super().__init__()

        """
        ASTConfig is used to define the architecture parameters of the model.
        Loading from pretrained ensures we use the same transformer structure
        as the original AudioSet fine-tuned model.
        """
        config = ASTConfig.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )

        # Update number of output labels
        config.num_labels = num_classes

        """
        ignore_mismatched_sizes=True allows replacing the classifier head
        even if the number of classes differs from the original AudioSet model.
        """
        self.ast = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            config=config,
            ignore_mismatched_sizes=True
        )

        self.time_frames = time_frames
        self.mel_bins = mel_bins

        """
        Original AST classifier:
            Linear(hidden_size → original_labels)

        New Custom Head:
            LayerNorm
            Linear(768 → 512)
            GELU activation
            Dropout(0.3)
            Linear(512 → num_classes)
        """
        hidden_size = self.ast.config.hidden_size  # Typically 768

        self.ast.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    @property
    def classifier(self):
        """
        Returns the classifier head of the AST model.
        """
        return self.ast.classifier

    def forward(self, input_values):
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
        if input_values.dim() == 4:
            input_values = input_values.squeeze(1)

        outputs = self.ast(input_values=input_values)

        return outputs.logits
    
