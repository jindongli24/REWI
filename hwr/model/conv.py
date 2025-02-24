import torch
import torch.nn as nn

__all__ = ['BLCNN']


class PatchEmbed(nn.Module):
    '''ConvNeXt-style patch embedding layer.

    Inputs:
        x (torch.Tensor): Input tensor (size_batch, num_chan, len_seq).
    Outputs:
        torch.Tensor: Output tensor (size_batch, num_chan, len_seq).
    '''

    def __init__(
        self, in_chan: int, out_chan: int, kernel: int = 2, stride: int = 2
    ) -> None:
        '''ConvNeXt-style patch embedding layer.

        Args:
            in_chan (int): Number of input channels.
            out_chan (int): Number of output channels.
            kernel (int, optional): Kernel size. Defaults to 2.
            stride (int, optional): Stride. Defaults to 2.
        '''
        super().__init__()

        self.conv = nn.Conv1d(in_chan, out_chan, kernel, stride)
        self.norm = nn.InstanceNorm1d(out_chan)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward method.

        Args:
            x (torch.Tensor): Input tensor (size_batch, num_chan, len_seq).

        Returns:
            torch.Tensor: Output tensor (size_batch, num_chan, len_seq).
        '''
        x = self.conv(x)
        x = self.norm(x)

        return x


class Conv(nn.Module):
    '''Convolutional module including depthwise convolutional layer,
    instance normalization, GELU activation function and dropout.

    Inputs:
        x (torch.Tensor): Input tensor (size_batch, num_chan, len_seq).
    Outputs:
        torch.Tensor: Output tensor (size_batch, num_chan, len_seq).
    '''

    def __init__(
        self,
        dim: int,
        kernel: int = 5,
        r_drop: float = 0.2,
    ) -> None:
        '''Convolutional module including depthwise convolutional layer,
        instance normalization, GELU activation function and dropout.

        Args:
            dim (int): Number of dimensions.
            kernel (int, optional): Kernel size. Defaults to 5.
            r_drop (float, optional): Dropping rate for dropout layer.
            Defaults to 0.2.
        '''
        super().__init__()

        self.dwconv = nn.Conv1d(
            dim, dim * 2, kernel, padding='same', groups=dim
        )
        self.pwconv = nn.Conv1d(dim * 2, dim, 1)
        self.norm = nn.InstanceNorm1d(dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(r_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward method.

        Args:
            x (torch.Tensor): Input tensor (size_batch, num_chan, len_seq).

        Returns:
            torch.Tensor: Output tensor (size_batch, num_chan, len_seq).
        '''
        x = self.dwconv(x)
        x = self.pwconv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)

        return x


class BLCNN(nn.Module):
    '''Convolutional baseline encoder.

    Inputs:
        x (torch.Tensor): Input tensor (size_batch, num_chan, len_seq).
    Outputs:
        torch.Tensor: Output tensor (size_batch, len_seq, num_chan).
    '''

    def __init__(
        self,
        in_chan: int,
        depths: list[int] = [3, 3, 3],
        dims: list[int] = [128, 256, 512],
    ) -> None:
        '''Convolutional baseline encoder.

        Args:
            in_chan (int): Number of input channels.
            depths (list[int]): Depths of all 3 blocks. Defaults to [3, 3, 3].
            dims (list[int]): Feature dimensions of all 3 blocks. 
            Defaults to [128, 256, 512].
        '''
        super().__init__()

        self.dims = [in_chan] + dims
        self.layers = nn.ModuleList([])

        for i in range(len(depths)):
            self.layers.append(PatchEmbed(self.dims[i], self.dims[i + 1]))
            self.layers.extend(
                [Conv(self.dims[i + 1]) for _ in range(depths[i])]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward method.

        Args:
            x (torch.Tensor): Input tensor (size_batch, num_chan, len_seq).

        Returns:
            torch.Tensor: Output tensor (size_batch, len_seq, num_chan).
        '''
        for layer in self.layers:
            x = layer(x)

        x = x.transpose(1, 2)

        return x

    @property
    def size_out(self) -> int:
        '''Get the number of output dimensions.

        Returns:
            int: Number of output dimensions.
        '''
        return self.dims[-1]
