import torch
import torch.nn as nn
from transformers import ASTForAudioClassification, ASTConfig


class ASTGenreClassifier(nn.Module):
    def __init__(self, num_classes: int, time_frames: int = 1024, mel_bins: int = 128):
        super().__init__()


        """
        It is used to instantiate an AST
        model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
        defaults will yield a similar configuration to that of the AST
        """
        config = ASTConfig.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        # config.hidden_dropout_prob = 0.1
        # config.attention_probs_dropout_prob = 0.1
        config.num_labels=num_classes

        self.ast = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            config=config,
            # num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

        # self.ast = ASTForAudioClassification.from_pretrained(
        #     "MIT/ast-finetuned-audioset-10-10-0.4593",
        #     num_labels=num_classes,
        #     ignore_mismatched_sizes=True
        # )

        self.time_frames = time_frames
        self.mel_bins = mel_bins

        hidden_size = self.ast.config.hidden_size  # 768
        self.ast.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    @property
    def classifier(self):
        return self.ast.classifier

    def forward(self, input_values):
        if input_values.dim() == 4:
            input_values = input_values.squeeze(1)

        outputs = self.ast(input_values=input_values)
        return outputs.logits
    