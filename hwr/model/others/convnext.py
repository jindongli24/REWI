# Liu et al. - 2020 - A ConvNet for the 2020s
# Modified from ConvNeXt (https://github.com/facebookresearch/ConvNeXt)

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_, DropPath

__all__ = ['ConvNeXt']


class LayerNorm(nn.Module):
    '''LayerNorm that supports two data formats: channels_last (default) or
    channels_first. The ordering of the dimensions in the inputs.
    channels_last corresponds to inputs with shape (batch_size, len_seq,
    channels) while channels_first corresponds to inputs with shape
    (batch_size, channels, len_seq).

    Inputs:
        x (torch.Tensor): Input tensor (batch_size, len_seq, channels) /
        (batch_size, channels, len_seq).
    Outputs:
        torch.Tensor: Input tensor (batch_size, len_seq, channels) /
        (batch_size, channels, len_seq).
    '''

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        data_format: str = "channels_last",
    ) -> None:
        '''LayerNorm that supports two data formats: channels_last (default)
        or channels_first. The ordering of the dimensions in the inputs.
        channels_last corresponds to inputs with shape (batch_size, len_seq,
        channels) while channels_first corresponds to inputs with shape
        (batch_size, channels, len_seq).

        Args:
            normalized_shape (int): Number of dimensions.
            eps (float, optional): Epsilon. Defaults to 1e-6.
            data_format (str, optional): Channel oreders. Options are "channel
            first" and "channel last". Defaults to "channels_last".

        Raises:
            NotImplementedError: Whether the given data format is valid.
        '''
        super().__init__()

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format

        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError

        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward method.

        Args:
            x (torch.Tensor): Input tensor (batch_size, len_seq, channels) /
            (batch_size, channels, len_seq).

        Returns:
            torch.Tensor: Input tensor (batch_size, len_seq, channels) /
            (batch_size, channels, len_seq).
        '''
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]

            return x


class Block(nn.Module):
    '''ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv;
    all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear
    -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch.

    Inputs:
        x (torch.Tensor): Input tensor (num_batch, num_chan, len_seq).
    Outputs:
        torch.Tensor: Output tensor(num_batch, num_chan, len_seq).
    '''

    def __init__(
        self,
        dim: int,
        ratio_rb: int = 4,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ) -> None:
        '''ConvNeXt Block. There are two equivalent implementations:
        (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU ->
        1x1 Conv; all in (N, C, H, W)
        (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) ->
        Linear -> GELU -> Linear; Permute back
        We use (2) as we find it slightly faster in PyTorch.

        Args:
            dim (int): Number of input channels.
            ratio_rb (int): Scale ratio of reversed bottleneck. Defaults to 4.
            drop_path (float): Stochastic depth rate. Default: 0.0
            layer_scale_init_value (float): Init value for Layer Scale.
            Default: 1e-6.
        '''
        super().__init__()

        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, ratio_rb * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(ratio_rb * dim, dim)
        self.gamma = (
            nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True
            )
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward method.

        Args:
            x (torch.Tensor): Input tensor (num_batch, num_chan, len_seq).

        Returns:
            torch.Tensor: Output tensor(num_batch, num_chan, len_seq).
        '''
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)  # (N, C, L) -> (N, L, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x

        x = x.permute(0, 2, 1)  # (N, L, C) -> (N, C, L)
        x = input + self.drop_path(x)

        return x


class ConvNeXt(nn.Module):
    '''ConvNeXt. A PyTorch impl of : `A ConvNet for the 2020s`
    (https://arxiv.org/pdf/2201.03545.pdf)
    '''

    def __init__(
        self,
        in_chans: int,
        depths: list[int] = [2, 2, 2],
        dims: list[int] = [96, 192, 384],
        ratio_rb: int = 3,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ) -> None:
        '''ConvNeXt. A PyTorch impl of : `A ConvNet for the 2020s`
        (https://arxiv.org/pdf/2201.03545.pdf)

        Args:
            in_chans (int): Number of input image channels. Default: 3
            depths (list[int]): Number of blocks at each stage.
            Default: [3, 3, 9, 3]
            dims (list[int]): Feature dimension at each stage.
            Default: [96, 192, 384, 768]
            ratio_rb (int): Scale ratio of reversed bottleneck. Defaults to 3.
            drop_path_rate (float): Stochastic depth rate. Default: 0.
            layer_scale_init_value (float): Init value for Layer Scale.
            Default: 1e-6.
        '''
        super().__init__()

        self.dims = dims
        self.num_stage = len(dims)

        self.downsample_layers = (
            nn.ModuleList()
        )  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv1d(in_chans, dims[0], kernel_size=2, stride=2),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)

        for i in range(self.num_stage - 1):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv1d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = (
            nn.ModuleList()
        )  # 3 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        cur = 0

        for i in range(self.num_stage):
            stage = nn.Sequential(
                *[
                    Block(
                        dim=dims[i],
                        ratio_rb=ratio_rb,
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        '''Initialize the weights of modules.

        Args:
            m (torch.nn.Module): Module to initialize.
        '''
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward method.

        Args:
            x (torch.Tensor): Input tensor (size_batch, num_chan, len_seq).

        Returns:
            torch.Tensor: Output tensor (size_batch, num_chan, len_seq).
        '''
        for i in range(self.num_stage):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        x = x.transpose(1, 2)
        x = self.norm(x)

        return x

    @property
    def size_out(self) -> int:
        '''Get the number of output dimensions.

        Returns:
            int: Number of output dimensions.
        '''
        return self.dims[-1]
