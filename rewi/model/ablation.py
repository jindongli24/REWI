import torch
import torch.nn as nn

__all__ = ['AblaDec', 'AblaEnc']


class PatchEmbed(nn.Module):
    '''Patch embedding layer.

    Inputs:
        x (torch.Tensor): Input tensor (size_batch, num_chan, len_seq).
    Outputs:
        torch.Tensor: Output tensor (size_batch, num_chan, len_seq).
    '''

    def __init__(
        self, in_chan: int, out_chan: int, inst: bool = False
    ) -> None:
        '''Patch embedding layer.

        Args:
            in_chan (int): Number of input channels.
            out_chan (int): Number of output channels.
            inst (bool, optional): Whether to use instance normalization. Defaults to False.
        '''
        super().__init__()

        self.conv = nn.Conv1d(in_chan, out_chan, 2, 2)
        self.norm = (
            nn.InstanceNorm1d(out_chan) if inst else nn.BatchNorm1d(out_chan)
        )

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
        ms: bool = False,
        inst: bool = False,
        gelu: bool = False,
    ) -> None:
        '''Multi-scale depth-dilated 1-D separable convolutional block.

        Args:
            dim (int): Number of dimension.
            ms (bool, optional): Whether to use multi-scale convolution. Defaults to False.
            inst (bool, optional): Whether to use instance normalization. Defaults to False.
            gelu (bool, optional): Whether to use GELU activation function. Defaults to False.
        '''
        super().__init__()

        self.dwconv1 = (
            nn.Conv1d(dim, dim * 2, 1, padding='same', groups=dim)
            if ms
            else None
        )
        self.dwconv3 = (
            nn.Conv1d(dim, dim * 2, 3, padding='same', groups=dim)
            if ms
            else None
        )
        self.dwconv5 = nn.Conv1d(dim, dim * 2, 5, padding='same', groups=dim)
        self.pwconv = nn.Conv1d(dim * 2, dim, 1)
        self.norm = nn.InstanceNorm1d(dim) if inst else nn.BatchNorm1d(dim)
        self.act = nn.GELU() if gelu else nn.ReLU()
        self.drop = nn.Dropout(0.3)
        self.fused = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward method.

        Args:
            x (torch.Tensor): Input tensor (size_batch, num_chan, len_seq).

        Returns:
            torch.Tensor: Output tensor (size_batch, num_chan, len_seq).
        '''
        if not self.fused and self.dwconv1 and self.dwconv3:
            x = self.dwconv1(x) + self.dwconv3(x) + self.dwconv5(x)
        else:
            x = self.dwconv5(x)

        x = self.pwconv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)

        return x

    def fuse(self) -> None:
        '''Add parameters of dwconv1 and dwconv3 to dwconv5.'''
        if self.dwconv1 and self.dwconv3:
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


class AblaEnc(nn.Module):
    '''Ablation encoder.

    Inputs:
        x (torch.Tensor): Input tensor (size_batch, num_chan, len_seq).
    Outputs:
        torch.Tensor: Output tensor (size_batch, len_seq, num_chan).
    '''

    def __init__(
        self,
        in_chan: int,
        rev: bool = False,
        embed: bool = False,
        sep: bool = False,
        ms: bool = False,
        inst: bool = False,
        gelu: bool = False,
    ) -> None:
        '''Ablation encoder.

        Args:
            in_chan (int): Number of input channel.
            rev (bool, optional): Whether to reverse channel order. Defaults to False.
            embed (bool, optional): Whether to use patch embedding layer. Defaults to False.
            sep (bool, optional): Whether to use depthwise separable convolution. Defaults to False.
            ms (bool, optional): Whether to use multi-scale convolution. Defaults to False.
            inst (bool, optional): Whether to use instance normalization. Defaults to False.
            gelu (bool, optional): Whether to use GELU activation function. Defaults to False.
        '''
        super().__init__()

        if rev:
            if embed:
                self.dim = [in_chan, 64, 128, 256]
            else:
                self.dim = [in_chan, 128, 256, 512]
        else:
            self.dim = [in_chan, 512, 256, 128]

        self.depths = [1, 1, 1]
        self.layers = nn.ModuleList([])

        for i in range(len(self.depths)):
            if embed:
                self.layers.append(
                    PatchEmbed(self.dim[i], self.dim[i + 1], inst)
                )

                if sep:
                    self.layers.append(MSConv(self.dim[i + 1], ms, inst, gelu))
                else:
                    self.layers.extend(
                        [
                            nn.Conv1d(
                                self.dim[i + 1],
                                self.dim[i + 1],
                                5 if i == 0 else 3,
                                padding='same',
                            ),
                            nn.BatchNorm1d(self.dim[i + 1]),
                            nn.ReLU(),
                            nn.Dropout(0.3),
                        ]
                    )
            else:
                self.layers.extend(
                    [
                        nn.Conv1d(
                            self.dim[i],
                            self.dim[i + 1],
                            5 if i == 0 else 3,
                            padding='same',
                        ),
                        nn.BatchNorm1d(self.dim[i + 1]),
                        nn.ReLU(),
                        nn.MaxPool1d(2, 2),
                        nn.Dropout(0.3),
                    ]
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
        return self.dim[-1]

    @property
    def ratio_ds(self) -> int:
        '''Get the downsample ratio between input length and output length.

        Returns:
            int: Downsample ratio.
        '''
        return 2 ** len(self.depths)


class AblaDec(nn.Module):
    '''Ablation decoder.

    Inputs:
        x (torch.Tensor): Input tensor (size_batch, len_seq, num_chan).
    Outputs:
        torch.Tensor: Output tensor of probabilities (size_batch, len_seq, num_cls).
    '''

    def __init__(
        self,
        size_in: int,
        num_cls: int,
        nohid: bool = False,
    ) -> None:
        '''Ablation decoder.

        Args:
            size_in (int): Number of input channel.
            num_cls (int): Number of categories.
            nohid (bool, optional): Whether to use extra linear layer with activation layer and dropout after LSTM. Defaults to False.
        '''
        super().__init__()

        self.lstm = nn.LSTM(
            size_in,
            64,
            2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )
        self.hid = nn.Sequential(
            nn.Linear(128, 100),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.fc = nn.Linear(128 if nohid else 100, num_cls)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Foward function.

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
