# He et al. - 2015 - Deep Residual Learning for Image Recognition
# Modified from vision (https://github.com/pytorch/vision)

import torch
import torch.nn as nn

__all__ = ['ResNet']


class BasicBlock(nn.Module):
    '''Basic block.

    Inputs:
        x (torch.Tensor): Input tensor (size_batch, num_chan, len_seq).
    Outputs:
        torch.Tensor: Output tensor (size_batch, num_chan, len_seq).
    '''

    def __init__(
        self,
        in_chan: int,
        dim: int,
        stride: int = 1,
        downsample: nn.Module = None,
    ) -> None:
        '''Basic block.

        Args:
            in_chan (int): Number of input channels.
            dim (int): Number of dimension of the block.
            stride (int, optional): Stide. Defaults to 1.
            downsample (torch.nn.Module, optional): Downsampling block.
            Defaults to None.
        '''
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_chan,
            dim,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            dim,
            dim,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(dim)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward method.

        Args:
            x (torch.Tensor): Input tensor (size_batch, num_chan, len_seq).

        Returns:
            torch.Tensor: Output tensor (size_batch, num_chan, len_seq).
        '''
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    '''ResNet.

    Inputs:
        x (torch.Tensor): Input tensor (size_batch, num_chan, len_seq)
    Outputs:
        torch.Tensor: Output tensor (size_batch, len_seq, num_chan)
    '''

    def __init__(
        self,
        in_chan: int,
        depths: list[int] = [3, 4, 6],
    ) -> None:
        '''ResNet.

        Args:
            in_chan (int): Number of input channels.
            depths (list[int], optional): Depths. Defaults to [3, 4, 5].
        '''
        super().__init__()

        inplanes = 64

        self.embed = nn.Sequential(
            nn.Conv1d(
                in_chan,
                inplanes,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm1d(inplanes),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([])

        for i, depth in enumerate(depths):
            planes = 64 * 2**i
            stride = 1 if i == 0 else 2
            downsample = nn.Sequential(
                nn.Conv1d(
                    inplanes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(planes),
            )
            layers = []

            for j in range(depth):
                layers.append(
                    BasicBlock(
                        inplanes,
                        planes,
                        stride if j == 0 else 1,
                        downsample if j == 0 else None,
                    )
                )
                inplanes = planes

            self.blocks.append(nn.Sequential(*layers))
            self.dim = planes

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward method.

        Args:
            x (torch.Tensor): Input tensor (size_batch, num_chan, len_seq)

        Returns:
            torch.Tensor: Output tensor (size_batch, len_seq, num_chan)
        '''
        x = self.embed(x)

        for block in self.blocks:
            x = block(x)

        x = x.transpose(1, 2)

        return x

    @property
    def size_out(self) -> int:
        '''Get the number of output dimensions.

        Returns:
            int: Number of output dimensions.
        '''
        return self.dim
