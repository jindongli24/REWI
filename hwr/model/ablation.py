import torch
import torch.nn as nn

__all__ = ['AblaDec', 'AblaEnc']


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
        revchan: bool = False,
        deeper: bool = False,
        embed: bool = False,
        sep: bool = False,
        inn: bool = False,
        gelu: bool = False,
        r_drop: bool = False,
    ) -> None:
        '''Ablation encoder.

        Args:
            in_chan (int): Number of input channels.
            revchan (bool, optional): Whether to use reversed channel sizes.
            Defaults to False.
            deeper (bool, optional): Whether to use 3x deeper network.
            Defaults to False.
            embed (bool, optional): Whether to use standalone patch embedding
            layer. Defaults to False.
            sep (bool, optional): Whether to use depth-dialated seperable
            depthwise convolution with kernel size of 5. Defaults to False.
            inn (bool, optional): Whether to use instance normalization
            instead of batch normalization. Defaults to False.
            gelu (bool, optional): Whether to use GELU activation layer
            instead of ReLU activation layer. Defaults to False.
            r_drop (bool, optional): Whether to use a dropout rate of 0.2
            instead of 0.3. Defaults to False.
        '''
        super().__init__()

        self.dim = (
            [in_chan, 128, 256, 512] if revchan else [in_chan, 512, 256, 128]
        )
        self.depth = 3 if deeper else 1
        self.embed = embed

        if embed:
            self.patch_embed = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv1d(self.dim[i], self.dim[i + 1], 2, 2),
                        (
                            nn.InstanceNorm1d(self.dim[i + 1])
                            if inn
                            else nn.BatchNorm1d(self.dim[i + 1])
                        ),
                    )
                    for i in range(3)
                ]
            )
        else:
            self.patch_embed = None

        self.blocks = nn.ModuleList([])

        # deeper network
        for i in range(3):
            layers = []

            for j in range(self.depth):
                dim_in = (
                    self.dim[i] if j == 0 and not embed else self.dim[i + 1]
                )
                dim_out = self.dim[i + 1]

                layers.append(
                    nn.Sequential(
                        nn.Conv1d(
                            dim_in,
                            dim_out * 2,
                            5,
                            padding='same',
                            groups=dim_in,
                        ),
                        nn.Conv1d(dim_out * 2, dim_out, 1),
                    )
                    if sep
                    else nn.Conv1d(
                        dim_in,
                        dim_out,
                        5 if i == 0 else 3,
                        padding='same',
                    )
                ),
                layers.append(
                    nn.InstanceNorm1d(dim_out)
                    if inn
                    else nn.BatchNorm1d(dim_out)
                ),
                layers.append(nn.GELU() if gelu else nn.ReLU()),

                if j == self.depth - 1 and not embed:
                    layers.append(nn.MaxPool1d(2, 2))

                layers.append(nn.Dropout(0.2 if r_drop else 0.3))

            self.blocks.append(nn.Sequential(*layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward method.

        Args:
            x (torch.Tensor): Input tensor (size_batch, num_chan, len_seq).

        Returns:
            torch.Tensor: Output tensor (size_batch, len_seq, num_chan).
        '''
        for i in range(3):
            if not self.patch_embed is None:
                x = self.patch_embed[i](x)

            x = self.blocks[i](x)

        x = x.transpose(1, 2)

        return x

    @property
    def size_out(self) -> int:
        '''Get the number of output dimensions.

        Returns:
            int: Number of output dimensions.
        '''
        return self.dim[-1]


class AblaDec(nn.Module):
    '''Ablation decoder.

    Inputs:
        x (torch.Tensor): Input tensor (size_batch, len_seq, num_chan).
    Outputs:
        torch.Tensor: Output tensor of probabilities (size_batch, len_seq,
        num_cls).
    '''

    def __init__(
        self,
        size_in: int,
        num_cls: int,
        wider: bool = False,
        deeper: bool = False,
        nohid: bool = False,
        r_drop: bool = False,
    ) -> None:
        '''Ablation decoder.

        Args:
            size_in (int): Number of input channel.
            num_cls (int): Number of categories.
            wider (bool, optional): Whether to use hidden size of 128 instead
            of 64 for LSTM. Defaults to False.
            deeper (bool, optional): Whether to use 3 layers of LSTM instead
            of 2 layers. Defaults to False.
            nohid (bool, optional): Whether to use extra linear layer with
            activation layer and dropout after LSTM. Defaults to False.
            r_drop (bool, optional): Whether to use a dropout rate of 0.2
            instead of 0.3. Defaults to False.
        '''
        super().__init__()

        size_hid = 128 if wider else 64
        num_layer = 3 if deeper else 2

        self.lstm = nn.LSTM(
            size_in,
            size_hid,
            num_layer,
            batch_first=True,
            dropout=0.2 if r_drop else 0.3,
            bidirectional=True,
        )
        self.hid = (
            None
            if nohid
            else nn.Sequential(
                nn.Linear(size_hid * 2, 100),
                nn.ReLU(),
                nn.Dropout(0.2 if r_drop else 0.3),
            )
        )
        self.fc = nn.Linear(size_hid * 2 if nohid else 100, num_cls)
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

        if not self.hid is None:
            x = self.hid(x)

        x = self.fc(x)
        x = self.softmax(x)

        return x
