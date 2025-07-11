import torch
import torch.nn as nn

__all__ = ['BLConv']


class PatchEmbed(nn.Module):
    '''Patch embedding layer.

    Inputs:
        x (torch.Tensor): Input tensor (size_batch, num_chan, len_seq).
    Outputs:
        torch.Tensor: Output tensor (size_batch, num_chan, len_seq).
    '''

    def __init__(
        self, in_chan: int, out_chan: int, kernel: int = 2, stride: int = 2
    ) -> None:
        '''Patch embedding layer.

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


class MSConv(nn.Module):
    '''Multi-scale depth-dilated 1-D separable convolutional block.

    Inputs:
        x (torch.Tensor): Input tensor (size_batch, num_chan, len_seq).
    Outputs:
        torch.Tensor: Output tensor (size_batch, num_chan, len_seq).
    '''

    def __init__(
        self,
        dim: int,
        r_drop: float = 0.2,
    ) -> None:
        '''Multi-scale depth-dilated 1-D separable convolutional block.

        Args:
            dim (int): Number of dimension.
            r_drop (float): Dropout rate. Defaults to 0.2.
        '''
        super().__init__()

        self.fused = False
        self.dwconv1 = nn.Conv1d(dim, dim * 2, 1, padding='same', groups=dim)
        self.dwconv3 = nn.Conv1d(dim, dim * 2, 3, padding='same', groups=dim)
        self.dwconv5 = nn.Conv1d(dim, dim * 2, 5, padding='same', groups=dim)
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
        if self.fused:
            x = self.dwconv5(x)
        else:
            x = self.dwconv1(x) + self.dwconv3(x) + self.dwconv5(x)

        x = self.pwconv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)

        return x

    def fuse(self) -> None:
        '''Add parameters of dwconv1 and dwconv3 to dwconv5.'''
        with torch.no_grad():
            self.dwconv5.weight.data[
                :, :, 2
            ] += self.dwconv1.weight.data.squeeze(-1)
            self.dwconv5.weight.data[:, :, 1:4] += self.dwconv3.weight.data
            self.dwconv5.bias.data += (
                self.dwconv1.bias.data + self.dwconv3.bias.data
            )

        del self.dwconv1
        del self.dwconv3
        self.fused = True


class BLConv(nn.Module):
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
            dims (list[int]): Feature dimensions of all 3 blocks. Defaults to [128, 256, 512].
        '''
        super().__init__()

        self.depths = depths
        self.dims = [in_chan] + dims
        self.layers = nn.ModuleList([])

        for i in range(len(depths)):
            self.layers.append(PatchEmbed(self.dims[i], self.dims[i + 1]))
            self.layers.extend(
                [MSConv(self.dims[i + 1]) for _ in range(depths[i])]
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

    def fuse(self) -> None:
        '''Fuse the convolutional layers.'''
        for m in self.layers:
            if hasattr(m, 'fuse'):
                m.fuse()

    @property
    def dim_out(self) -> int:
        '''Get the number of output dimensions.

        Returns:
            int: Number of output dimensions.
        '''
        return self.dims[-1]

    @property
    def ratio_ds(self) -> int:
        '''Get the downsample ratio between input length and output length.

        Returns:
            int: Downsample ratio.
        '''
        return 2 ** len(self.depths)
