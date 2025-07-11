# Ott et al. - 2022 - Benchmarking Online Sequence-to-Sequence and Character-based Handwriting Recognition from IMU-Enhanced Pens

import torch
import torch.nn as nn

__all__ = ['OttBiLSTM', 'OttCNN']


class OttCNN(nn.Module):
    '''Ott's convolutional encoder.

    Inputs:
        x (torch.Tensor): Input tensor (size_batch, num_chan, len_seq).
    Outputs:
        torch.Tensor: Output tensor (size_batch, len_seq, num_chan).
    '''

    def __init__(self, in_chan: int) -> None:
        '''Ott's convolutional encoder.

        Args:
            in_chan (int): Number of input channels.
        '''
        super().__init__()

        self.conv1 = nn.Conv1d(in_chan, 200, 4, padding='same')
        self.mp1 = nn.MaxPool1d(2, 2)
        self.bn1 = nn.BatchNorm1d(200)
        self.do1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv1d(200, 200, 4, padding='same')
        self.mp2 = nn.MaxPool1d(2, 2)
        self.bn2 = nn.BatchNorm1d(200)
        self.do2 = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward method.

        Args:
            x (torch.Tensor): Input tensor (size_batch, num_chan, len_seq).

        Returns:
            torch.Tensor: Output tensor (size_batch, len_seq, num_chan).
        '''
        x = self.conv1(x)
        x = self.mp1(x)
        x = self.bn1(x)
        x = self.do1(x)
        x = self.conv2(x)
        x = self.mp2(x)
        x = self.bn2(x)
        x = self.do2(x)

        x = x.transpose(1, 2)

        return x

    @property
    def dim_out(self) -> int:
        '''Get the number of output dimensions.

        Returns:
            int: Number of output dimensions.
        '''
        return 200

    @property
    def ratio_ds(self) -> int:
        '''Get the downsample ratio between input length and output length.

        Returns:
            int: Downsample ratio.
        '''
        return 4


class OttBiLSTM(nn.Module):
    '''Ott's Bi-LSTM module for classification.

    Inputs:
        x (torch.Tensor): Input tensor (size_batch, len_seq, num_chan).
    Outputs:
        torch.Tensor: Output tensor of probabilities (size_batch, len_seq, num_cls).
    '''

    def __init__(self, size_in: int, num_cls: int) -> None:
        '''Ott's Bi-LSTM module for classification.

        Args:
            size_in (int): Number of input channel.
            num_cls (int): Number of categories.
        '''
        super().__init__()

        self.lstm = nn.LSTM(
            size_in, 60, 2, batch_first=True, bidirectional=True
        )
        self.hid = nn.Linear(120, 100)
        self.fc = nn.Linear(100, num_cls)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward method.

        Args:
            x (torch.Tensor): Input tensor (size_batch, len_seq, num_chan).

        Returns:
            torch.Tensor: Output tensor of probabilities (size_batch, len_seq, num_cls).
        '''
        x, _ = self.lstm(x)
        x = self.hid(x)
        x = self.fc(x)
        x = self.softmax(x)

        return x
