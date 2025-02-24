# Liu et al. - 2021 - Swin Transformer V2: Scaling Up Capacity and Resolution
# Modified from Swin-Transformer (https://github.com/microsoft/Swin-Transformer)

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple, trunc_normal_

__all__ = ['SwinTransformerV2']


class Mlp(nn.Module):
    '''Multilayer perceptron for SwinTransformerBlock1D.

    Inputs:
        x (torch.Tensor): Input tensor (size_batch, len_sequence, num_channel).
    Outputs:
        torch.Tensor: Output tensor (size_batch, len_sequence, num_channel).
    '''

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        '''Multilayer perceptron for Swin Transformers.

        Args:
            in_features (int): Number of dimension of input features.
            hidden_features (int, optional): Number of dimension of hidden
            features. If not given, in_features will be used. Defaults to None.
            out_features (int, optional): Number of dimension of output
            features. If not given, in_features will be used. Defaults to None.
            act_layer (torch.nn.Module, optional): Activation layer. Defaults
            to torch.nn.GELU.
            drop (float, optional): Drop rate for torch.nn.Dropout. Defaults
            to 0.0.
        '''
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward method.

        Args:
            x (torch.Tensor): Input tensor (size_batch, len_sequence,
            num_channel).

        Returns:
            torch.Tensor: Output tensor (size_batch, len_sequence,
            num_channel).
        '''
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    '''Window patitioning for SwinTransformerBlock1D.

    Args:
        x (torch.Tensor): Input tensor (size_batch, len_sequence, num_channel).
        window_size (int): Window size.

    Returns:
        torch.Tensor: Output tensor (num_window * size_batch, size_window,
        num_channel).
    '''
    B, L, C = x.shape
    x = x.view(B, L // window_size, window_size, C)
    windows = x.contiguous().view(-1, window_size, C)

    return windows


def window_reverse(windows, window_size, L):
    '''Window patition reversing for SwinTransformerBlock1D.

    Args:
        x (torch.Tensor): Input tensor (num_window * size_batch, size_window,
        num_channel).
        window_size (int): Window size.
        L (int): Sequence length.

    Returns:
        torch.Tensor: Output tensor (size_batch, len_sequence, num_channel).
    '''
    B = int(windows.shape[0] / (L / window_size))
    x = windows.view(B, L // window_size, window_size, -1)
    x = x.contiguous().view(B, L, -1)
    return x


class WindowAttention(nn.Module):
    '''Window based multi-head self attention (W-MSA) module with relative
    position bias. It supports both of shifted and non-shifted window.

    Inputs:
        x (torch.Tensor): Input tensor (num_window * size_batch, num_window,
        num_channel).
        mask (torch.Tensor, optional): (0/-100) mask with shape of
        (num_window, size_window, size_window). Defaults to None.
    Outputs:
        torch.Tensor: Output tensor (num_window * size_batch, num_window,
        num_channel).
    '''

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        pretrained_window_size: int = 0,
    ) -> None:
        '''Window based multi-head self attention (W-MSA) module with relative
        position bias. It supports both of shifted and non-shifted window.

        Args:
            dim (int): Number of input dimension.
            window_size (int): Window size.
            num_heads (int): Number of attention heads.
            qkv_bias (bool, optional): Whether to add a learnable bias to
            query, key and value. Defaults to True.
            attn_drop (float, optional): Dropout ratio of attention weight.
            Defaults to 0.0.
            proj_drop (float, optional): Dropout ratio of output. Defaults to
            0.0.
            pretrained_window_size (int, optional): Window size in
            pre-training. Defaults to 0.
        '''
        super().__init__()

        self.dim = dim
        self.window_size = window_size  # Wl
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True
        )

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(1, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )

        # get relative_coords_table
        relative_coords_l = torch.arange(
            -(self.window_size - 1),
            self.window_size,
            dtype=torch.float32,
        )
        relative_coords_table = (
            torch.stack(torch.meshgrid([relative_coords_l], indexing='ij'))
            .permute(1, 0)
            .contiguous()
            .unsqueeze(0)
        )  # 1, 2*Wl-1, 1

        if pretrained_window_size > 0:
            relative_coords_table[:, :, :] /= pretrained_window_size - 1
        else:
            relative_coords_table[:, :, :] /= self.window_size - 1

        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = (
            torch.sign(relative_coords_table)
            * torch.log2(torch.abs(relative_coords_table) + 1.0)
            / np.log2(8)
        )
        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_l = torch.arange(self.window_size)
        coords = torch.stack(
            torch.meshgrid([coords_l], indexing='ij')
        )  # 1, Wl
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 1, Wl, Wl
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wl, Wl, 1
        relative_coords[:, :, 0] += (
            self.window_size - 1
        )  # shift to start from 0
        relative_position_index = relative_coords.sum(-1)  # Wl, Wl
        self.register_buffer(
            "relative_position_index", relative_position_index
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        '''Forward method.

        Args:
            x (torch.Tensor): Input tensor (num_window * size_batch,
            size_window, num_channel).
            mask (torch.Tensor, optional): (0/-100) mask with shape of
            (num_window, size_window, size_window). Defaults to None.

        Returns:
            torch.Tensor: Output tensor (num_window * size_batch, num_window, num_channel).
        '''
        B_, N, C = x.shape
        qkv_bias = None

        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias,
                )
            )

        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(
            2, 0, 3, 1, 4
        )  # 3, num_window * size_batch, num_head, size_window, dim_embed
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(
            -2, -1
        )
        logit_scale = torch.clamp(
            self.logit_scale,
            max=torch.log(torch.tensor(1.0 / 0.01, device=attn.device)),
        ).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(
            self.relative_coords_table
        ).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size,
            self.window_size,
            -1,
        )  # Wl,Wl,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wl, Wl
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(
                B_ // nW, nW, self.num_heads, N, N
            ) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class SwinTransformerBlock(nn.Module):
    '''Swin Transformer Block.

    Inputs:
        x (torch.Tensor): Input tensor (size_batch, len_sequence, num_channel).
    Outputs:
        torch.Tensor: Output tensor (size_batch, len_sequence, num_channel).
    '''

    def __init__(
        self,
        dim: int,
        input_resolution: int,
        num_heads: int,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        pretrained_window_size: int = 0,
    ) -> None:
        '''Swin Transformer Block.

        Args:
            dim (int): Number of input channels.
            input_resolution (int): Input resolution.
            num_heads (int): Number of attention heads.
            window_size (int, optional): Window size. Defaults to 8.
            shift_size (int, optional): Shift size for SW-MSA. Defaults to 0.
            mlp_ratio (float, optional): Ratio of MLP hidden dimensions to
            embedding dimensions. Defaults to 4.0.
            qkv_bias (bool, optional): Whether to add a learnable bias to
            query, key and value. Defaults to True.
            drop (float, optional): Dropout rate. Defaults to 0.0.
            attn_drop (float, optional): Dropout rate of attention weights.
            Defaults to 0.0.
            drop_path (float, optional): Stochastic depth rate. Defaults to
            0.0.
            act_layer (torch.nn.Module, optional): Activation layer. Defaults
            to torch.nn.GELU.
            norm_layer (torch.nn.Module, optional): Normalization layer.
            Defaults to torch.nn.LayerNorm.
            pretrained_window_size (int, optional): Windows size in
            pre-training. Defaults to 0.
        '''
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if self.input_resolution <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = self.input_resolution

        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            pretrained_window_size=pretrained_window_size,
        )
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            seq_mask = torch.zeros((1, input_resolution, 1))  # 1 L 1
            segs = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0

            for d in segs:
                seq_mask[:, d, :] = cnt
                cnt += 1

            mask_windows = window_partition(
                seq_mask, self.window_size
            )  # nW, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer('attn_mask', attn_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward method.

        Args:
            x (torch.Tensor): Input tensor (size_batch, len_sequence,
            num_channel).

        Returns:
            torch.Tensor: Output tensor (size_batch, len_sequence,
            num_channel).
        '''
        B, L, C = x.shape
        shortcut = x

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size), dims=(1))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nW*B, window_size, C
        x_windows = x_windows.view(
            -1, self.window_size, C
        )  # nW*B, window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(
            x_windows, mask=self.attn_mask
        )  # nW*B, window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, C)
        shifted_x = window_reverse(
            attn_windows, self.window_size, L
        )  # B, L, C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size),
                dims=(1),
            )
        else:
            x = shifted_x

        x = x.view(B, L, C)
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x


class PatchMerge(nn.Module):
    '''Patch Merging Layer.

    Inputs:
        x (torch.Tensor): Input tensor (size_batch, len_sequence,
        num_channel).
    Outputs:
        torch.Tensor: Output tensor (size_batch, len_sequence / 2,
        num_channel * 2).'''

    def __init__(self, dim: int, norm_layer: nn.Module = nn.LayerNorm) -> None:
        '''Patch merging layer.

        Args:
            dim (int): Number of input channels.
            norm_layer (torch.nn.Module, optional): Normalization layer.
            Defaults to torch.nn.LayerNorm.
        '''
        super().__init__()

        # self.dim = dim
        self.reduction = nn.Linear(2 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward method.

        Args:
            x (torch.Tensor): Input tensor (size_batch, len_sequence,
            num_channel).

        Returns:
            torch.Tensor: Output tensor (size_batch, len_sequence / 2,
            num_channel * 2).
        '''
        x0 = x[:, 0::2, :]  # B L/2 C
        x1 = x[:, 1::2, :]  # B L/2 C
        x = torch.cat([x0, x1], -1)  # B L/2 2*C

        x = self.reduction(x)
        x = self.norm(x)

        return x


class BasicLayer(nn.Module):
    '''A basic Swin Transformer layer for one stage.

    Inputs:
        x (torch.Tensor): Input tensor (size_batch, len_sequence, num_channel).
    Outputs:
        torch.Tensor: Output tensor (size_batch, len_sequence, num_channel).
    '''

    def __init__(
        self,
        dim: int,
        input_resolution: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        downsample: nn.Module = None,
        pretrained_window_size: int = 0,
    ) -> None:
        '''A basic Swin Transformer layer for one stage.

        Args:
            dim (int): Number of input channels.
            input_resolution (int): Input resolution.
            depth (int): Number of blocks.
            num_heads (int): Number of attention heads.
            window_size (int): Window size.
            mlp_ratio (float, optional): Ratio of MLP hidden dimensions to
            embedding dimensions. Defaults to 4.0.
            qkv_bias (bool, optional): Whether to add a learnable bias to
            query, key and value. Defaults to True.
            drop (float, optional): Dropout rate. Defaults to 0.0.
            attn_drop (float, optional): Dropout rate of attention weights.
            Defaults to 0.0.
            drop_path (float, optional): Stochastic depth rate. Defaults to
            0.0.
            norm_layer (torch.nn.Module, optional): Normalization layer.
            Defaults to torch.nn.LayerNorm.
            downsample (torch.nn.Module, optional): Downsample layer at the
            end of the layer. Defaults to None.
            pretrained_window_size (int, optional): Window size in
            pre-training. Defaults to 0.
        '''
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=(
                        drop_path[i]
                        if isinstance(drop_path, list)
                        else drop_path
                    ),
                    norm_layer=norm_layer,
                    pretrained_window_size=pretrained_window_size,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def _init_respostnorm(self) -> None:
        '''Initialize normalization layers in SwinTransformerBlock1D.'''
        # for blk in self.blocks:
        #     nn.init.constant_(blk.norm1.bias, 0)
        #     nn.init.constant_(blk.norm1.weight, 0)
        #     nn.init.constant_(blk.norm2.bias, 0)
        #     nn.init.constant_(blk.norm2.weight, 0)
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward methods.

        Args:
            x (torch.Tensor): Input tensor (size_batch, len_sequence,
            num_channel).

        Returns:
            torch.Tensor: Output tensor (size_batch, len_sequence,
            num_channel).
        '''
        for blk in self.blocks:
            x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


class PatchEmbed(nn.Module):
    '''Patch embedding layer.

    Inputs:
        x (torch.Tensor): Input tensor (size_batch, len_sequence, num_channel).
    Outputs:
        torch.Tensor: Output tensor (size_batch, len_sequence, num_channel).
    '''

    def __init__(
        self,
        seq_len: int,
        patch_size: int,
        in_chans: int,
        embed_dim: int,
        norm_layer: nn.Module = None,
    ) -> None:
        '''Patch embedding layer.

        Args:
            len_seq (int): Length of sequences.
            patch_size (int, optional): Patch token size.
            in_chans (int, optional): Number of input channels.
            embed_dim (int, optional): Number of linear projection output
            channels.
            norm_layer (torch.nn.Module, optional): Normalization layer.
            Defaults to None.
        '''
        super().__init__()

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_patches = math.ceil(seq_len / patch_size)
        self.patches_resolution = math.ceil(seq_len / patch_size)

        self.proj = nn.Conv1d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward method.

        Args:
            x (torch.Tensor): Input tensor (size_batch, num_channel,
            len_sequence).

        Returns:
            torch.Tensor: Output tensor (size_batch, len_sequence,
            num_channel).
        '''
        x = self.proj(x).transpose(1, 2)  # B, L, C

        if self.norm is not None:
            x = self.norm(x)

        return x


class SwinTransformerV2(nn.Module):
    '''Swin Transformer V2.

    Inputs:
        x (torch.Tensor): Input tensor (size_batch, num_channel, len_sequence).
    Outputs:
        torch.Tensor: Output tensor (size_batch, len_sequence, num_channel).
    '''

    def __init__(
        self,
        in_chans,
        seq_len: int = 1024,
        patch_size: int = 2,
        embed_dim: int = 96,
        depths: list[int] = [2, 2, 2],
        num_heads: list[int] = [3, 6, 12],
        window_size: int = 8,
        mlp_ratio: float = 2.5,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        patch_norm: bool = True,
        pretrained_window_sizes: list[int] = [0, 0, 0, 0],
    ) -> None:
        '''Swin Transformer V2.

        Args:
            in_chans (_type_): Number of input channels.
            seq_len (int, optional): Length of sequences. Defaults to 256.
            patch_size (int, optional): Patch size. Defaults to 2.
            embed_dim (int, optional): Patch embedding dimension. Defaults to
            96.
            depths (list[int], optional): Depth of each Swin Transformer
            layer. Defaults to [2, 2, 2].
            num_heads (list[int], optional): Number of attention heads in
            different layers. Defaults to [3, 6, 12].
            window_size (int, optional): Window size. Defaults to 8.
            mlp_ratio (float, optional): Ratio of MLP hidden dimensions to
            embedding dimensions. Defaults to 4.0.
            qkv_bias (bool, optional): Whether to add a learnable bias to
            query, key and value. Defaults to True.
            drop_rate (float, optional): Dropout rate. Defaults to 0.0.
            attn_drop_rate (float, optional): Dropout rate of attention
            weights. Defaults to 0.0.
            drop_path_rate (float, optional): Stochastic depth rate. Defaults
            to 0.1.
            norm_layer (torch.nn.Module, optional): Normalization layer.
            Defaults to torch.nn.LayerNorm.
            patch_norm (bool, optional): Whether to add normalization after
            patch embedding. Defaults to True.
            pretrained_window_sizes (list[int], optional): Pretrained window
            sizes of each layer. Defaults to [0, 0, 0, 0].
        '''
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            seq_len=seq_len,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=math.ceil(patches_resolution // (2**i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                ],
                norm_layer=norm_layer,
                downsample=(
                    PatchMerge if (i_layer < self.num_layers - 1) else None
                ),
                pretrained_window_size=pretrained_window_sizes[i_layer],
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)

        self.apply(self._init_weights)

        for bly in self.layers:
            bly._init_respostnorm()

    def _init_weights(self, m: nn.Module) -> None:
        '''Initialize weights of layers.

        Args:
            m (torch.nn.Module): Layers to initialize.
        '''
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)

            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward method.

        Args:
            x (torch.Tensor): Input tensor (size_batch, num_channel,
            len_sequence).

        Returns:
            torch.Tensor: Output tensor (size_batch, len_sequence,
            num_channel).
        '''
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C

        return x

    @property
    def size_out(self) -> int:
        '''Get the number of output dimensions.

        Returns:
            int: Number of output dimensions.
        '''
        return self.embed_dim * 2 ** (self.num_layers - 1)
