# Webbi et al. - 2021 - Towards an IMU-based Pen Online Handwriting  Recognizer

import torch
import torch.nn as nn

__all__ = ['MohDnc', 'MohEec']


class MohEnc(nn.Module):
    '''Mohamad's convolutional encoder.

    Inputs:
        x (torch.Tensor): Input tensor (size_batch, num_chan, len_seq).
    Outputs:
        torch.Tensor: Output tensor (size_batch, len_seq, num_chan).
    '''

    def __init__(self, in_chan: int) -> None:
        '''Mohamad's convolutional encoder.

        Args:
            in_chan (int): Number of input channels.
        '''
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv1d(in_chan, 512, 5, padding='same'),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Dropout(0.3),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(512, 256, 3, padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Dropout(0.3),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(256, 128, 3, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Dropout(0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward method.

        Args:
            x (torch.Tensor): Input tensor (size_batch, num_chan, len_seq).

        Returns:
            torch.Tensor: Output tensor (size_batch, len_seq, num_chan).
        '''
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.transpose(1, 2)

        return x

    @property
    def size_out(self) -> int:
        '''Get the number of output dimensions.

        Returns:
            int: Number of output dimensions.
        '''
        return 128


class MohDec(nn.Module):
    '''Mohamad's Bi-LSTM decoder.

    Inputs:
        x (torch.Tensor): Input tensor (size_batch, len_seq, num_chan).
    Outputs:
        torch.Tensor: Output tensor of probabilities (size_batch, len_seq,
        num_cls).
    '''

    def __init__(self, size_in: int, num_cls: int) -> None:
        '''Mohamad's Bi-LSTM decoder.

        Args:
            size_in (int): Number of input channel.
            num_cls (int): Number of categories.
        '''
        super().__init__()

        self.lstm = nn.LSTM(
            size_in, 64, 2, batch_first=True, dropout=0.3, bidirectional=True
        )
        self.hid = nn.Sequential(
            nn.Linear(128, 100),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.fc = nn.Linear(100, num_cls)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Foward function.

        Args:
            x (torch.Tensor): Input tensor (size_batch, len_seq, num_chan).

        Returns:
            torch.Tensor: Output tensor of probabilities (size_batch, len_seq,
            num_cls).
        '''
        x, _ = self.lstm(x)
        x = self.hid(x)
        x = self.fc(x)
        x = self.softmax(x)

        return x
