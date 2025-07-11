# Dosovitskiy et al. - 2021 - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
# Modified from vit-pytorch (https://github.com/lucidrains/vit-pytorch)

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

__all__ = ['ViT']


class FeedForward(nn.Module):
    '''Feed forward network.

    Inputs:
        x (torch.Tensor): Input tensor (size_batch, num_patch, num_dim).
    Outputs:
        torch.Tensor: Output tensor (size_batch, num_patch, num_dim).
    '''

    def __init__(
        self, dim: int, hidden_dim: int, dropout: float = 0.0
    ) -> None:
        '''Feed forward network.

        Args:
            dim (int): Number of input and output dimensions.
            hidden_dim (int): Number of hidden dimensions.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
        '''
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward method.

        Args:
            x (torch.Tensor): Input tensor (size_batch, num_patch, num_dim).

        Returns:
            torch.Tensor: Output tensor (size_batch, num_patch, num_dim).
        '''
        return self.net(x)


class Attention(nn.Module):
    '''Multi-head attention layer.

    Inputs:
        x (torch.Tensor): Input tensor (size_batch, num_patch, num_dim).
    Outputs:
        torch.Tensor: Output tensor (size_batch, num_patch, num_dim).
    '''

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ) -> None:
        '''Multi-head attention layer.

        Args:
            dim (int): Number of input dimension.
            heads (int, optional): Number of heads. Defaults to 8.
            dim_head (int, optional): Number of head dimension. Defaults to 64.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
        '''
        super().__init__()

        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward method.

        Args:
            x (torch.Tensor): Input tensor (size_batch, num_patch, num_dim).

        Returns:
            torch.Tensor: Output tensor (size_batch, num_patch, num_dim).
        '''
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class Transformer(nn.Module):
    '''Transformer block.

    Inputs:
        x (torch.Tensor): Input tensor (size_batch, num_patch, num_dim).
    Outputs:
        torch.Tensor: Output tensor (size_batch, num_patch, num_dim).
    '''

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        '''Transformer block.

        Args:
            dim (int): Number of input dimension.
            depth (int): Depth.
            heads (int): Number of heads.
            dim_head (int): Number of head dimension.
            mlp_dim (int): Number of hidden dimension of MLP.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
        '''
        super().__init__()

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout,
                        ),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward method.

        Args:
            x (torch.Tensor): Input tensor (size_batch, num_patch, num_dim).

        Returns:
            torch.Tensor: Output tensor (size_batch, num_patch, num_dim).
        '''
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return x


# class TransEnc(nn.Module):
class ViT(nn.Module):
    '''Transformer encoder.

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
        dim: int = 128,
        depth: int = 4,
        heads: int = 8,
        mlp_dim: int = 512,
        dim_head: int = 128,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ) -> None:
        '''Transformer Encoder.

        Args:
            in_chan (int): Number of input channels.
            len_seq (int, optional): Length of input. Defaults to 1024.
            patch_size (int, optional): Patch size. Defaults to 8.
            dim (int, optional): Number of patch dimension. Defaults to 128.
            depth (int, optional): Depth. Defaults to 4.
            heads (int, optional): Number of heads. Defaults to 8.
            mlp_dim (int, optional): Number of hidden dimensions of feed forward network. Defaults to 512.
            dim_head (int, optional): Number of head dimensions. Defaults to 128.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
            emb_dropout (float, optional): Dropout rate of patch embedding layer. Defaults to 0.0.
        '''
        super().__init__()

        assert (
            len_seq % patch_size
        ) == 0, 'Input length cannot be divided by patch size.'

        self.dim = dim
        self.patch_size = patch_size

        num_patches = len_seq // patch_size
        patch_dim = in_chan * patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward method.

        Args:
            x (torch.Tensor): Input tensor (size_batch, num_chan, len_seq).

        Returns:
            torch.Tensor: Output tensor (size_batch, num_patch, num_dim).
        '''
        x = self.to_patch_embedding(x)

        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)

        return x

    @property
    def dim_out(self) -> int:
        '''Get the number of output dimensions.

        Returns:
            int: Number of output dimensions.
        '''
        return self.dim

    @property
    def ratio_ds(self) -> int:
        '''Get the downsample ratio between input length and output length.

        Returns:
            int: Downsample ratio.
        '''
        return self.patch_size