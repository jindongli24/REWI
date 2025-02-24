# Tolstikhin et al. - 2021 - MLP-Mixer: An all-MLP Architecture for Vision
# Modified from mlp-mixer-pytorch (https://github.com/lucidrains/mlp-mixer-pytorch)

import torch
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange

__all__ = ['MLPMixer']


class PreNormResidual(nn.Module):
    '''Wrapper for layer normalization and residual connection.'''

    def __init__(self, dim: int, fn: nn.Sequential) -> None:
        '''Wrapper for layer normalization and residual connection.

        Args:
            dim (int): Input dimension.
            fn (nn.Sequential): Mixer block to warp.
        '''
        super().__init__()

        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward method.

        Args:
            x (torch.Tensor): Input tensor (size_batch, num_patch, num_dim).

        Returns:
            torch.Tensor: Output tensor (size_batch, num_patch, num_dim).
        '''
        return self.fn(self.norm(x)) + x


def mlp(
    dim: int,
    expansion_factor: float = 4.0,
    dropout: float = 0.0,
    dense: nn.Module = nn.Linear,
) -> nn.Sequential:
    '''Build MLP block.

    Args:
        dim (int): Number of input and output dimensions.
        expansion_factor (float, optional): Expansion factor for hidden size.
        Defaults to 4.0.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        dense (torch.nn.Module, optional): Dense layer module.
        Defaults to torch.nn.Linear.

    Returns:
        nn.Sequential: MLP block.
    '''
    inner_dim = int(dim * expansion_factor)

    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout),
    )


class MLPMixer(nn.Module):
    '''MLP Mixer.

    Inputs:
        x (torch.Tensor): Input tensor (size_batch, num_chan, len_seq).
    Outputs:
        torch.Tensor: Output tensor (size_batch, num_patch, num_dim).
    '''

    def __init__(
        self,
        in_chan: int,
        len_seq: int = 1024,
        patch_size: int = 8,
        dim: int = 512,
        depth: int = 6,
        expansion_factor: float = 4.0,
        expansion_factor_token: float = 0.5,
        dropout: float = 0.0,
    ) -> None:
        '''MLP Mixer.

        Args:
            in_chan (int): Number of input channels.
            len_seq (int, optional): Length of input sequences.
            Defaults to 1024.
            patch_size (int, optional): Patch size. Defaults to 8.
            dim (int, optional): Dimension. Defaults to 512.
            depth (int, optional): Depth. Defaults to 6.
            expansion_factor (float, optional): Expansion factor for token
            mixing. Defaults to 4.0.
            expansion_factor_token (float, optional): Expansion factor for
            channel mixing. Defaults to 0.5.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
        '''
        super().__init__()

        self.dim = dim
        num_patches = len_seq // patch_size
        chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear

        self.patch_embed = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p=patch_size),
            nn.Linear(patch_size * in_chan, dim),
        )
        self.layers = nn.Sequential(
            *[
                nn.Sequential(
                    PreNormResidual(
                        dim,
                        mlp(
                            num_patches, expansion_factor, dropout, chan_first
                        ),
                    ),
                    PreNormResidual(
                        dim,
                        mlp(dim, expansion_factor_token, dropout, chan_last),
                    ),
                )
                for _ in range(depth)
            ],
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward method.

        Args:
            x (torch.Tensor): Input tensor (size_batch, num_chan, len_seq).

        Returns:
            torch.Tensor: Output tensor (size_batch, num_patch, num_dim).
        '''
        x = self.patch_embed(x)
        x = self.layers(x)
        x = self.norm(x)

        return x

    @property
    def size_out(self) -> int:
        '''Get the number of output dimensions.

        Returns:
            int: Number of output dimensions.
        '''
        return self.dim
