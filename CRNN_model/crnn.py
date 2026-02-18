import torch
import torch.nn as nn
import torch.nn.functional as F
from CRNN_model.attention import AudioMultiHeadAttention


"""
reference to model

https://github.com/GitYCC/crnn-pytorch/blob/master/src/model.py
"""


class CRNN(nn.Module):
    def __init__(
            self, num_class: int, channel: int = 1, height: int = 128, width: int = 216, linear_layer_out: int = 64, rnn_hidden: int = 128,
        ):
        super(CRNN, self).__init__()

        self.cnn, (output_channel, output_height, output_width) = self.cnn_backbone(
            channel=channel, height=height, width=width
        )

        self.conv_to_fc = nn.Linear(
            in_features=output_channel * output_height,
            out_features=linear_layer_out,
            bias=True
        )

        self.lstm = nn.LSTM(linear_layer_out, rnn_hidden, bidirectional=True)
        self.lstm_dropout = nn.Dropout(0.3)

        self.attention = AudioMultiHeadAttention(
            n_embeddings=rnn_hidden*2,
            n_heads=4,
            head_size=rnn_hidden // 2,
        )

        self.head = nn.Linear(
            in_features=2 * rnn_hidden,
            out_features=num_class
        )

    def cnn_backbone(self, channel: int, height: int, width: int):

        channels     = [channel, 16, 32, 64, 128, 128]
        kernel_sizes = [3, 3, 3, 3, 3]
        strids       = [1, 1, 1, 1, 1]
        paddings     = [1, 1, 1, 1, 1]

        cnn = nn.Sequential()

        def conv_layer(idx: int, batch_norm: bool = False):
            input_channel  = channels[idx]
            output_channel = channels[idx + 1]
            cnn.add_module(
                f"conv{idx}", 
                nn.Conv2d(input_channel, output_channel, kernel_sizes[idx], strids[idx], paddings[idx]))
            
            if batch_norm:
                cnn.add_module(f"batchnorm{idx}", nn.BatchNorm2d(output_channel))

            cnn.add_module(f"relu{idx}", nn.LeakyReLU(0.2, inplace=True))

        # [1,height,width]
        conv_layer(0)
        cnn.add_module("polling0", nn.MaxPool2d(kernel_size=2, stride=2))
        # [16,height/2,width/2 ]

        conv_layer(1)
        cnn.add_module("polling1", nn.MaxPool2d(kernel_size=2, stride=2))
        # [32,height/4,width/4 ]

        conv_layer(2)
        # [64,height/4,width/4 ]

        conv_layer(3, batch_norm=True)
        cnn.add_module("polling2", nn.MaxPool2d(kernel_size=2, stride=2))
        # [128,height/8,width/8 ]

        conv_layer(4, batch_norm=True)
        cnn.add_module("polling3", nn.MaxPool2d(kernel_size=2, stride=2))
        # [128, height/16, width/16]

        output_channel = channels[-1]
        output_height  = height // 16
        output_width   = width  // 16

        return cnn, (output_channel, output_height, output_width)
    

    def trainable_params(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = total - trainable
        print(f"  Total     : {total:,}")
        print(f"  Trainable : {trainable:,}")
        print(f"  Frozen    : {frozen:,}")


    def forward(self, x: torch.Tensor):
        conv = self.cnn(x)                                

        batch, channel, height, width = conv.size()
        conv = conv.view(batch, channel * height, width)
        conv = conv.permute(2, 0, 1)

        seq = self.conv_to_fc(conv)

        recurrent, _ = self.lstm(seq)
        recurrent = self.lstm_dropout(recurrent)

        recurrent = recurrent.permute(1, 0, 2)

        context   = self.attention(recurrent)
        context   = context.mean(dim=1)

        return self.head(context)
            