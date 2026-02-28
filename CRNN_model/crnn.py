import torch
import torch.nn as nn
import torch.nn.functional as F
from CRNN_model.attention import AudioMultiHeadAttention


"""
CRNN Model for Audio / Spectrogram Classification

Reference:
https://github.com/GitYCC/crnn-pytorch/blob/master/src/model.py

Architecture Summary:
    Input Spectrogram
        ↓
    CNN Backbone (feature extraction)
        ↓
    Reshape to Sequence (Width → Time steps)
        ↓
    Linear Projection
        ↓
    2-Layer BiLSTM
        ↓
    Multi-Head Self Attention
        ↓
    Fully Connected Layers
        ↓
    Classification Head
"""


class CRNN(nn.Module):
    """
    CRNN (Convolutional Recurrent Neural Network)

    Designed for spectrogram-based classification.

    Combines:
        - CNN for spatial feature extraction
        - BiLSTM for temporal modeling
        - Multi-Head Attention for contextual weighting
        - Fully connected classifier head

    Expected Input Shape:
        [Batch, Channel, Height, Width]

    Typical Example:
        [B, 1, 128, 216]

    Output:
        [Batch, num_class]
    """

    def __init__(
        self,
        num_class: int,
        channel: int = 1,
        height: int = 128,
        width: int = 216,
        linear_layer_out: int = 64,
        rnn_hidden: int = 128,
    ):
        """
        Args:
            num_class (int): Number of output classes.
            channel (int): Input channels (1 = grayscale spectrogram).
            height (int): Spectrogram height (frequency bins).
            width (int): Spectrogram width (time frames).
            linear_layer_out (int): Output size after CNN-to-sequence projection.
            rnn_hidden (int): Hidden size of LSTM.
        """
        super(CRNN, self).__init__()

        self.cnn, (output_channel, output_height, output_width) = self.cnn_backbone(
            channel=channel, height=height, width=width
        )

        """
        CNN Output Shape:
            [B, C, H, W]

        We reshape it to:
            [W, B, C*H]

        Then apply linear projection to reduce feature dimension.
        """
        self.conv_to_fc = nn.Linear(
            in_features=output_channel * output_height,
            out_features=linear_layer_out,
            bias=True
        )

        """
        Input  : [Time, Batch, Features]
        Output : [Time, Batch, 2 * rnn_hidden]
        """
        self.lstm = nn.LSTM(
            linear_layer_out,
            rnn_hidden,
            num_layers=2,
            bidirectional=True,
            dropout=0.3
        )

        self.lstm_dropout = nn.Dropout(0.3)

        """
        Helps model focus on important time steps.

        Embedding size = 2 * rnn_hidden (because BiLSTM)
        """
        self.attention = AudioMultiHeadAttention(
            n_embeddings=rnn_hidden * 2,
            n_heads=8,
            head_size=rnn_hidden // 2,
        )

        self.fc1 = nn.Linear(
            in_features=2 * rnn_hidden,
            out_features=rnn_hidden
        )

        self.fc1_dropout = nn.Dropout(0.3)

        self.head = nn.Linear(
            in_features=rnn_hidden,
            out_features=num_class
        )

    def cnn_backbone(self, channel: int, height: int, width: int):
        """
        Builds CNN feature extractor.

        Structure:
            Conv → ReLU → Pool
            Conv → ReLU → Pool
            ...
            6 pooling stages total

        Downsampling Factor:
            Height  // 64
            Width   // 64

        Returns:
            cnn (nn.Sequential)
            output dimensions tuple
        """

        channels = [channel, 16, 32, 64, 64, 128, 128, 256]
        kernel_sizes = [3] * 8
        strides = [1] * 8
        paddings = [1] * 8

        cnn = nn.Sequential()

        def conv_layer(idx: int, batch_norm: bool = False):
            input_channel = channels[idx]
            output_channel = channels[idx + 1]

            cnn.add_module(
                f"conv{idx}",
                nn.Conv2d(input_channel, output_channel,
                          kernel_sizes[idx], strides[idx], paddings[idx])
            )

            if batch_norm:
                cnn.add_module(f"batchnorm{idx}",
                               nn.BatchNorm2d(output_channel))

            cnn.add_module(f"relu{idx}",
                           nn.LeakyReLU(0.2, inplace=True))

        # Block 1
        conv_layer(0)
        cnn.add_module("pool0", nn.MaxPool2d(2, 2))

        # Block 2
        conv_layer(1)
        cnn.add_module("pool1", nn.MaxPool2d(2, 2))

        # Block 3
        conv_layer(2)

        conv_layer(3, batch_norm=True)
        cnn.add_module("pool2", nn.MaxPool2d(2, 2))

        # Block 4
        conv_layer(4, batch_norm=True)
        cnn.add_module("pool3", nn.MaxPool2d(2, 2))

        # Block 5
        conv_layer(5, batch_norm=True)
        cnn.add_module("pool4", nn.MaxPool2d(2, 2))

        # Block 6
        conv_layer(6, batch_norm=True)
        cnn.add_module("pool5", nn.MaxPool2d(2, 2))

        output_channel = channels[-1]
        output_height = height // 64
        output_width = width // 64

        return cnn, (output_channel, output_height, output_width)

    def trainable_params(self):
        """
        Prints total, trainable, and frozen parameters.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        frozen = total - trainable

        print(f"  Total     : {total:,}")
        print(f"  Trainable : {trainable:,}")
        print(f"  Frozen    : {frozen:,}")

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Args:
            x (Tensor): [B, C, H, W]

        Returns:
            logits (Tensor): [B, num_class]
        """

        # CNN feature extraction
        conv = self.cnn(x)
        # Shape: [B, C, H', W']

        batch, channel, height, width = conv.size()

        # Convert to sequence
        conv = conv.view(batch, channel * height, width)
        conv = conv.permute(2, 0, 1)
        # Shape: [W', B, C*H']

        # Linear projection
        seq = self.conv_to_fc(conv)

        # BiLSTM
        recurrent, _ = self.lstm(seq)
        recurrent = self.lstm_dropout(recurrent)

        # Convert to batch-first
        recurrent = recurrent.permute(1, 0, 2)
        # Shape: [B, W', 2*rnn_hidden]

        # Attention
        context = self.attention(recurrent)

        # Average over time dimension
        context = context.mean(dim=1)

        # Fully connected classifier
        x = self.fc1(context)
        x = F.leaky_relu(x, 0.2)
        x = self.fc1_dropout(x)

        return self.head(x)
    