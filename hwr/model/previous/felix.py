# Ott et al. - 2022 - Benchmarking Online Sequence-to-Sequence and Character-based Handwriting Recognition from IMU-Enhanced Pens

import torch
import torch.nn as nn

__all__ = ['FelBiLSTM', 'FelCNN', 'FelInceptionTime']


class InceptionModule(nn.Module):
    def __init__(
        self, ni: int, nf: int, ks: int = 40, bottleneck: bool = True
    ) -> None:
        super().__init__()

        ks = [ks // (2**i) for i in range(3)]
        ks = [k if k % 2 != 0 else k - 1 for k in ks]  # ensure odd ks
        bottleneck = bottleneck if ni > 1 else False

        self.bottleneck = (
            nn.Conv1d(ni, nf, 1, padding='same', bias=False)
            if bottleneck
            else nn.Sequential()
        )
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    nf if bottleneck else ni, nf, k, padding='same', bias=False
                )
                for k in ks
            ]
        )
        self.maxconvpool = nn.Sequential(
            *[
                nn.MaxPool1d(3, stride=1, padding=1),
                nn.Conv1d(ni, nf, 1, padding='same', bias=False),
            ]
        )
        self.bn = nn.BatchNorm1d(nf * 4)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_tensor = x
        x = self.bottleneck(input_tensor)
        x = torch.cat(
            [l(x) for l in self.convs] + [self.maxconvpool(input_tensor)], 1
        )

        return self.act(self.bn(x))


class FelInceptionTime(nn.Module):
    def __init__(
        self,
        ni: int,
        nf: int = 96,
        residual: bool = True,
        depth: int = 11,
    ) -> None:
        super().__init__()

        self.nf, self.residual, self.depth = nf, residual, depth
        self.inception, self.shortcut = nn.ModuleList(), nn.ModuleList()
        self.act = nn.ReLU()

        for d in range(depth):
            self.inception.append(
                InceptionModule(ni if d == 0 else nf * 4, nf)
            )

            if self.residual and d % 3 == 2:
                n_in, n_out = ni if d == 2 else nf * 4, nf * 4
                self.shortcut.append(
                    nn.BatchNorm1d(n_in)
                    if n_in == n_out
                    else nn.Sequential(
                        nn.Conv1d(
                            n_in,
                            n_out,
                            1,
                            padding='same',
                        ),
                        nn.BatchNorm1d(n_out),
                    )
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x

        for d, l in enumerate(range(self.depth)):
            x = self.inception[d](x)

            if self.residual and d % 3 == 2:
                res = x = self.act(x.add(self.shortcut[d // 3](res)))

        x = x.transpose(1, 2)

        return x

    @property
    def size_out(self) -> int:
        '''Get the number of output dimensions.

        Returns:
            int: Number of output dimensions.
        '''
        return self.nf * 4


class FelCNN(nn.Module):
    '''Felix's convolutional encoder.

    Inputs:
        x (torch.Tensor): Input tensor (size_batch, num_chan, len_seq).
    Outputs:
        torch.Tensor: Output tensor (size_batch, len_seq, num_chan).
    '''

    def __init__(self, in_chan: int) -> None:
        '''Felix's convolutional encoder.

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
    def size_out(self) -> int:
        '''Get the number of output dimensions.

        Returns:
            int: Number of output dimensions.
        '''
        return 200


class FelBiLSTM(nn.Module):
    '''Felix's Bi-LSTM module for classification.

    Inputs:
        x (torch.Tensor): Input tensor (size_batch, len_seq, num_chan).
    Outputs:
        torch.Tensor: Output tensor of probabilities (size_batch, len_seq,
        num_cls).
    '''

    def __init__(self, size_in: int, num_cls: int) -> None:
        '''Felix's Bi-LSTM module for classification.

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
            torch.Tensor: Output tensor of probabilities (size_batch, len_seq,
            num_cls).
        '''
        x, _ = self.lstm(x)
        x = self.hid(x)
        x = self.fc(x)
        x = self.softmax(x)

        return x
