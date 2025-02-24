# Beck et al. - 2024 - xLSTM: Extended Long Short-Term Memory
# Modified from xlstm (https://github.com/NX-AI/xlstm)

import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['mLSTM']


def bias_linspace_init_(
    param: torch.Tensor, start: float = 3.4, end: float = 6.0
) -> torch.Tensor:
    '''Linearly spaced bias init across dimensions.

    Args:
        param (torch.Tensor): Parameters of bias to initialize.
        start (float, optional): Start of the linear space. Defaults to 3.4.
        end (float, optional): End of the linear space. Defaults to 6.0.

    Returns:
        torch.Tensor: Initialized parameters.
    '''
    n_dims = param.shape[0]
    init_vals = torch.linspace(start, end, n_dims)

    with torch.no_grad():
        param.copy_(init_vals)

    return param


def small_init_init_(param: torch.Tensor, dim: int) -> torch.Tensor:
    '''Fills the input Tensor with values according to the method described in
    Transformers without Tears: Improving the Normalization of Self-Attention,
    using a normal distribution. Adopted from gpt-neox from EleutherAI.

    Args:
        param (torch.Tensor): Parameters to initialize.
        dim (int): Number of dimensions of the parameter tensor.

    Returns:
        torch.Tensor: Initialized parameters.
    '''
    std = math.sqrt(2 / (5 * dim))
    torch.nn.init.normal_(param, mean=0.0, std=std)

    return param


def wang_init_(param: torch.Tensor, dim: int, num_layers: int):
    '''Wang initialization method adopted from gpt-neox from EleutherAI.

    Args:
        param (torch.Tensor): Parameters to initialize.
        dim (int): Number of dimensions of the parameter tensor.
        num_layers (int): Number of layers in the network.

    Returns:
        torch.Tensor: Initialized parameters.
    '''
    std = 2 / num_layers / math.sqrt(dim)
    torch.nn.init.normal_(param, mean=0.0, std=std)

    return param


def conv1d_step(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv1d_bias: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    '''Single-step inference for 1-D causal convolution.

    B: batch size
    S: sequence length
    D: feature dimension
    KS: kernel size

    Args:
        x (torch.Tensor): Input tensor (B, S, H).
        conv_state (torch.Tensor): Convolutional hidden state (B, KS, H).
        conv1d_weight (torch.Tensor): Weight parameters (KS, H).
        conv1d_bias (torch.Tensor): Bias parameters (1, H).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Tensors of result and next hidden state.
    '''
    conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=1))
    conv_state[:, -1:, :] = x
    y = torch.sum(conv_state * conv1d_weight, dim=1, keepdim=True)

    if conv1d_bias is not None:
        y += conv1d_bias

    return y, conv_state


def parallel_stabilized_simple(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
    lower_triangular_matrix: torch.Tensor = None,
    stabilize_rowwise: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    '''This is the mLSTM cell in parallel form. This version is stabilized. We
    control the range of exp() arguments by ensuring that they are always
    smaller than 0.0 by subtracting the maximum.

    Args:
        queries (torch.Tensor): Queries tensor (B, NH, S, DH).
        keys (torch.Tensor): Keys tensor (B, NH, S, DH).
        values (torch.Tensor): Values tensor (B, NH, S, DH).
        igate_preact (torch.Tensor): Input gate pre-activation tensor
        (B, NH, S, 1).
        fgate_preact (torch.Tensor): Forget gate pre-activation tensor
        (B, NH, S, 1).
        lower_triangular_matrix (torch.Tensor, optional): Causal mask (S,S).
        Defaults to None.
        stabilize_rowwise (bool, optional): Wether to stabilize the
        combination matrix C rowwise (take maximum per row). Alternative:
        Subtract the maximum over all rows. Defaults to True.
        eps (float, optional): Epsilon. Defaults to 1e-6.

    Returns:
        torch.Tensor: Hidden state tensor (B, NH, S, DH).
    '''

    B, NH, S, DH = queries.shape
    _dtype, _device = queries.dtype, queries.device

    # forget gate matrix
    log_fgates = F.logsigmoid(fgate_preact)  # (B, NH, S, 1)

    if lower_triangular_matrix is None or S < lower_triangular_matrix.size(-1):
        ltr = torch.tril(torch.ones((S, S), dtype=torch.bool, device=_device))
    else:
        ltr = lower_triangular_matrix

    log_fgates_cumsum = torch.cat(
        [
            torch.zeros((B, NH, 1, 1), dtype=_dtype, device=_device),
            torch.cumsum(log_fgates, dim=-2),
        ],
        dim=-2,
    )  # (B, NH, S+1, 1)
    # for each batch/head this is a matrix of shape (S+1, S+1) containing the
    # cumsum of the log forget gate values in the second dimension (colum
    # dimension). Each row has the same is a copy of the first row. First
    # entry of each row is zero.
    rep_log_fgates_cumsum = log_fgates_cumsum.repeat(
        1, 1, 1, S + 1
    )  # (B, NH, S+1, S+1)
    # Now in each row cut off / subtract the forgetgate values of the later
    # timesteps where col j > row i
    _log_fg_matrix = rep_log_fgates_cumsum - rep_log_fgates_cumsum.transpose(
        -2, -1
    )  # (B, NH, S+1, S+1)
    # Causal masking & selection of the correct submatrix, such that
    # forgetgate at timestep t is not applied to the input at timestep t
    log_fg_matrix = torch.where(
        ltr, _log_fg_matrix[:, :, 1:, 1:], -float("inf")
    )  # (B, NH, S, S)

    # gate decay matrix D (combination of forget gate and input gate)
    log_D_matrix = log_fg_matrix + igate_preact.transpose(
        -2, -1
    )  # (B, NH, S, S)

    # D matrix stabilization
    if stabilize_rowwise:
        max_log_D, _ = torch.max(
            log_D_matrix, dim=-1, keepdim=True
        )  # (B, NH, 1, 1)
    else:
        max_log_D = torch.max(
            log_D_matrix.view(B, NH, -1), dim=-1, keepdim=True
        )[0].unsqueeze(-1)
        # (B, NH, S, 1)

    log_D_matrix_stabilized = log_D_matrix - max_log_D  # (B, NH, S, S)
    D_matrix = torch.exp(log_D_matrix_stabilized)  # (B, NH, S, S)

    keys_scaled = keys / math.sqrt(DH)

    # combination matrix C
    qk_matrix = queries @ keys_scaled.transpose(-2, -1)  # (B, NH, S, S)
    C_matrix = qk_matrix * D_matrix  # (B, NH, S, S)
    normalizer = torch.maximum(
        C_matrix.sum(dim=-1, keepdim=True).abs(), torch.exp(-max_log_D)
    )  # (B, NH, S, 1)
    C_matrix_normalized = C_matrix / (normalizer + eps)  # (B, NH, S, S)

    # retrieved values
    h_tilde_state = C_matrix_normalized @ values  # (B, NH, S, DH)

    return h_tilde_state


def recurrent_step_stabilized_simple(
    c_state: torch.Tensor,
    n_state: torch.Tensor,
    m_state: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    '''This is a single step of the mLSTM operation in recurrent form.

    Args:
        c_state (torch.Tensor): Cell state tensor (B, NH, DH, DH).
        n_state (torch.Tensor): Normalizer state tensor (B, NH, DH, 1).
        m_state (torch.Tensor): Stabalizer state tensor (B, NH, 1, 1).
        q (torch.Tensor): Queries tensor (B, NH, 1, DH).
        k (torch.Tensor): Keys tensor (B, NH, 1, DH).
        v (torch.Tensor): Values tensor (B, NH, 1, DH).
        igate_preact (torch.Tensor): Input gate pre-activation tensor
        (B, NH, 1, 1).
        fgate_preact (torch.Tensor): Forget gate pre-activation tensor
        (B, NH, 1, 1).

    Returns:
        tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]: (hidden_state
        [B, NH, DH], (c_state_new [B, NH, DH, DH], n_state_new
        [B, NH, DH, 1]], m_state_new [B, NH, 1, 1]))
    '''
    B, NH, S, DH = q.shape
    # projections
    q, k, v = (
        q.squeeze_(2).unsqueeze(-1),
        k.squeeze_(2).unsqueeze(-1),
        v.squeeze_(2).unsqueeze(-1),
    )  # (B, NH, DH, 1)

    # gates
    log_fg_act = F.logsigmoid(fgate_preact)  # (B, NH, 1, 1)

    # update rule
    m_state_new = torch.max(
        log_fg_act + m_state, igate_preact
    )  # (B, NH, 1, 1)

    fg_act = torch.exp(log_fg_act + m_state - m_state_new)  # (B, NH, 1, 1)
    ig_act = torch.exp(igate_preact - m_state_new)  # (B, NH, 1, 1)

    k_scaled = k / math.sqrt(DH)

    c_state_new = fg_act * c_state + ig_act * (
        k_scaled @ v.transpose(-1, -2)
    )  # (B, NH, DH, DH)
    n_state_new = fg_act * n_state + ig_act * k_scaled  # (B, NH, DH, 1)

    h_num = q.transpose(-1, -2) @ c_state_new  # (B, NH, 1, DH)

    qn_dotproduct = q.transpose(-1, -2) @ n_state_new  # (B, NH, 1, 1)
    max_val = torch.exp(-m_state_new)  # (B, NH, 1, 1)
    h_denom = torch.maximum(qn_dotproduct.abs(), max_val) + eps
    h = h_num / h_denom  # (B, NH, 1, DH) / (B, NH, 1, 1) = (B, NH, 1, DH)

    return h, (c_state_new, n_state_new, m_state_new)


class LayerNorm(nn.Module):
    '''LayerNorm  with an optional bias as PyTorch doesn't support simply
    bias=False.

    Inputs:
        input (torch.Tensor): Input tensor (B, NH, S, DH).
    Outputs:
        torch.Tensor: Output tensor (B, NH, S, DH).
    '''

    def __init__(
        self,
        ndim: int = -1,
        weight: bool = True,
        bias: bool = False,
        eps: float = 1e-5,
        residual_weight: bool = True,
    ) -> None:
        '''Layer normalization layer with optional bias.

        Args:
            ndim (int, optional): Number of input dimensions. Defaults to -1.
            weight (bool, optional): Whether to use weight parameters.
            Defaults to True.
            bias (bool, optional): Whether to use bias parameters. Defaults to
            False.
            eps (float, optional): Epsilon. Defaults to 1e-5.
            residual_weight (bool, optional): Whether to use residual weight.
            Defaults to True.
        '''
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(ndim)) if weight else None
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps
        self.residual_weight = residual_weight
        self.ndim = ndim
        self.reset_parameters()

    @property
    def weight_proxy(self) -> torch.Tensor:
        '''Get weight parameters.

        Returns:
            torch.Tensor: Weight parameter tensor.
        '''
        if self.weight is None:
            return None

        if self.residual_weight:
            return 1.0 + self.weight
        else:
            return self.weight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        '''Forward funciton.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        '''
        return F.layer_norm(
            input,
            normalized_shape=(self.ndim,),
            weight=self.weight_proxy,
            bias=self.bias,
            eps=self.eps,
        )

    def reset_parameters(self) -> None:
        '''Reset parameters.'''
        if self.weight_proxy is not None:
            if self.residual_weight:
                nn.init.zeros_(self.weight)
            else:
                nn.init.ones_(self.weight)

        if self.bias is not None:
            nn.init.zeros_(self.bias)


class MultiHeadLayerNorm(LayerNorm):
    '''LayerNorm for multi-head attention with an optional bias.

    Inputs:
        input (torch.Tensor): Input tensor (B, NH, S, DH).
    Outputs:
        torch.Tensor: Output tensor (B, NH, S, DH).
    '''

    def __init__(
        self,
        ndim: int = -1,
        weight: bool = True,
        bias: bool = False,
        eps: float = 1e-5,
        residual_weight: bool = True,
    ) -> None:
        '''Layer normalization for multi-head attention layer with optional
        bias.

        Args:
            ndim (int, optional): Number of input dimensions. Defaults to -1.
            weight (bool, optional): Whether to use weight parameters.
            Defaults to True.
            bias (bool, optional): Whether to use bias parameters. Defaults to
            False.
            eps (float, optional): Epsilon. Defaults to 1e-5.
            residual_weight (bool, optional): Whether to use residual weight.
            Defaults to True.
        '''
        super().__init__(ndim, weight, bias, eps, residual_weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        '''Forward method.

        Args:
            input (torch.Tensor): Input tensor (B, NH, S, DH).

        Returns:
            torch.Tensor: Output tensor (B, NH, S, DH).
        '''
        B, NH, S, DH = input.shape

        # (B, S, NH, DH)
        gn_in_1 = input.transpose(1, 2)
        # (B * S, NH * DH)
        gn_in_2 = gn_in_1.reshape(B * S, NH * DH)
        out = F.group_norm(
            gn_in_2,
            num_groups=NH,
            weight=self.weight_proxy,
            bias=self.bias,
            eps=self.eps,
        )
        # (B * S), (NH * DH) -> (B, S, NH, DH) -> (B, NH, S, DH)
        out = out.view(B, S, NH, DH).transpose(1, 2)

        return out


class LinearHeadwiseExpand(nn.Module):
    '''This is a structured projection layer that projects the input to a
    higher dimension. It only allows integer up-projection factors, i.e. the
    output dimension is a multiple of the input dimension.

    Inputs:
        x (torch.Tensor): Input tensor (B, S, H).
    Outputs:
        torch.Tensor: Output tensor (B, S, H).
    '''

    def __init__(
        self,
        in_features: int,
        num_heads: int,
        expand_factor_up: float = 1,
        bias: bool = True,
        trainable_weight: bool = True,
        trainable_bias: bool = True,
    ) -> None:
        '''LinearHeadwiseExpand module.

        Args:
            in_features (int): Number of input dimensions.
            num_heads (int): Number of heads.
            expand_factor_up (float, optional): Factor for up expansion.
            Defaults to 1.
            bias (bool, optional): Whether to use bias. Defaults to True.
            trainable_weight (bool, optional): Whether to use trainable weight
            parameters. Defaults to True.
            trainable_bias (bool, optional): Whether to use trainable bias
            parameters. Only applied when bias is True. Defaults to True.
        '''
        super().__init__()

        self.in_features = in_features
        self.num_heads = num_heads
        self.expand_factor_up = expand_factor_up
        self.bias = bias
        self.trainable_weight = trainable_weight
        self.trainable_bias = trainable_bias
        self.out_features = round(expand_factor_up * in_features)

        out_features_per_head = self.out_features // num_heads
        self.weight = nn.Parameter(
            torch.empty(
                num_heads, out_features_per_head, in_features // num_heads
            ),
            requires_grad=trainable_weight,
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_features),
                requires_grad=trainable_bias,
            )
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        '''Reset parameters with small_init.'''
        nn.init.normal_(
            self.weight.data,
            mean=0.0,
            std=math.sqrt(2 / 5 / self.weight.shape[-1]),
        )

        if self.bias is not None:
            nn.init.zeros_(self.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward method.

        Args:
            x (torch.Tensor): Input tensor (B, S, H).

        Returns:
            torch.Tensor: Output tensor (B, S, H).
        '''
        shape = x.shape
        x = x.view(*shape[:-1], self.num_heads, -1)
        x = torch.einsum("...hd,hod->...ho", x, self.weight)
        x = x.reshape(*shape[:-1], -1)

        if self.bias is not None:
            x = x + self.bias

        return x


class CausalConv1d(nn.Module):
    '''Implements causal depthwise convolution of a time series tensor.

    Inputs:
        x (torch.Tensor): Input tensor.
        conv_state (torch.Tensor, optional): Convolution state (B, S, H).
        Defaults to None.
        return_last_state (bool, optional): Whether to return the last
        convolution state. Defaults to False.
    Outputs:
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Output tensor
        (B, S, H) if return_last_state is False. Otherwise, a tuple of output
        tensor and convolution state (B, KS - 1, H).
    '''

    def __init__(
        self,
        feature_dim: int,
        kernel_size: int = 4,
        causal_conv_bias: bool = True,
        channel_mixing: bool = False,
    ) -> None:
        '''1-D causal convolution.

        Args:
            feature_dim (int): Number of features in the input tensor
            kernel_size (int, optional): Size of the kernel for the depthwise
            convolution. Defaults to 4.
            causal_conv_bias (bool, optional): Whether to use bias in the
            depthwise convolution. Defaults to True.
            channel_mixing (bool, optional): Whether to use channel mixing
            (i.e. groups=1) or not (i.e. groups=feature_dim). If True, it
            mixes the convolved features across channels. If False, all the
            features are convolved independently.
        '''
        super().__init__()

        self.feature_dim = feature_dim
        self.kernel_size = kernel_size
        self.causal_conv_bias = causal_conv_bias
        self.channel_mixing = channel_mixing
        self.groups = 1 if channel_mixing else feature_dim

        if kernel_size:
            # padding of this size assures temporal causality
            self.pad = kernel_size - 1
            self.conv = nn.Conv1d(
                in_channels=feature_dim,
                out_channels=feature_dim,
                kernel_size=kernel_size,
                padding=self.pad,
                groups=self.groups,
                bias=causal_conv_bias,
            )
        else:
            self.conv = None  # Noop

        # B, C, L
        self.reset_parameters()

    def reset_parameters(self) -> None:
        '''Reset parameters'''
        self.conv.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor = None,
        return_last_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        '''Forward method.

        Args:
            x (torch.Tensor): Input tensor (B, S, H).
            conv_state (torch.Tensor, optional): Convolution state
            (B, KS - 1, H). Defaults to None.
            return_last_state (bool, optional): Whether to return the last
            convolution state. Defaults to False.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Output tensor
            (B, S, H) if return_last_state is False. Otherwise, a tuple of
            output tensor and convolution state (B, KS - 1, H).
        '''
        if conv_state is not None:
            x = torch.cat([conv_state, x], dim=1)

        if self.kernel_size == 0:
            return x

        # (B,F,T) tensor - now in the right shape for conv layer
        y = x.transpose(2, 1)
        y = self.conv(y)  # (B,F,T+pad) tensor

        if conv_state is not None:
            y = y[:, :, conv_state.shape[1] :]

        if return_last_state:
            return y[:, :, : -self.pad].transpose(2, 1), x[:, -self.pad :]
        else:
            return y[:, :, : -self.pad].transpose(2, 1)

    def step(
        self,
        x: torch.Tensor,
        conv_state: tuple[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor]]:
        '''Single-step operation for recurrent form.

        Args:
            x (torch.Tensor): Input tensor.
            conv_state (tuple[torch.Tensor], optional): Previous convolution
            states. Defaults to None.

        Returns:
            tuple[torch.Tensor, tuple[torch.Tensor]]: Output
        '''
        # TODO: test input setting
        if self.kernel_size == 0:
            return x, conv_state

        B, _, D = x.shape

        if conv_state is None:
            conv_state = (
                torch.zeros(
                    size=(B, self.kernel_size, D),
                    device=self.conv.weight.device,
                    dtype=self.conv.weight.dtype,
                ),
            )

        y, conv_state = conv1d_step(
            x,
            conv_state[0],
            # rearrange(, "D 1 KS -> KS D")
            self.conv.weight[:, 0, :].transpose(0, 1),
            conv1d_bias=(self.conv.bias if self.causal_conv_bias else None),
        )

        return y, (conv_state,)


class mLSTMCell(nn.Module):
    '''mLSTM cell.

    Inputs:
        q (torch.Tensor): Queries tensor (B, S, H).
        k (torch.Tensor): Keys tensor (B, S, H).
        v (torch.Tensor): Values tensor (B, S, H).
    Outputs:
        torch.Tensor: Output tensor.
    '''

    def __init__(
        self, context_length: int, embedding_dim: int, num_heads: int
    ) -> None:
        '''mLSTM cell.

        Args:
            context_length (int): Length of the input sequence.
            embedding_dim (int): Number of input dimensions.
            num_heads (int): Number of heads.
        '''
        super().__init__()

        self.context_length = context_length
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        self.backend_fn = parallel_stabilized_simple
        self.backend_fn_step = recurrent_step_stabilized_simple

        self.igate = nn.Linear(3 * embedding_dim, num_heads)
        self.fgate = nn.Linear(3 * embedding_dim, num_heads)

        self.outnorm = MultiHeadLayerNorm(
            ndim=embedding_dim, weight=True, bias=False
        )

        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(
                    context_length,
                    context_length,
                    dtype=torch.bool,
                )
            ),
            persistent=False,
        )

        self.reset_parameters()

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        '''Forward method.

        Args:
            q (torch.Tensor): Queries tensor (B, S, H).
            k (torch.Tensor): Keys tensor (B, S, H).
            v (torch.Tensor): Values tensor (B, S, H).

        Returns:
            torch.Tensor: Output tensor.
        '''
        B, S, _ = q.shape  # (B, S, H)

        if_gate_input = torch.cat([q, k, v], dim=-1)
        q = q.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)
        k = k.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)
        v = v.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)

        q = q.transpose(1, 2)  # (B, NH, S, DH)
        k = k.transpose(1, 2)  # (B, NH, S, DH)
        v = v.transpose(1, 2)  # (B, NH, S, DH)

        # compute input and forget gate pre-activations
        igate_preact = self.igate(if_gate_input)  # (B, S, NH)
        igate_preact = igate_preact.transpose(-1, -2).unsqueeze(
            -1
        )  # (B, NH, S, 1)
        fgate_preact = self.fgate(if_gate_input)  # (B, S, NH)
        fgate_preact = fgate_preact.transpose(-1, -2).unsqueeze(
            -1
        )  # (B, NH, S, 1)

        h_state = self.backend_fn(
            queries=q,
            keys=k,
            values=v,
            igate_preact=igate_preact,
            fgate_preact=fgate_preact,
            lower_triangular_matrix=self.causal_mask,
        )  # (B, NH, S, DH)

        h_state_norm = self.outnorm(h_state)  # (B, NH, S, DH)
        h_state_norm = h_state_norm.transpose(1, 2).reshape(
            B, S, -1
        )  # (B, NH, S, DH) -> (B, S, NH, DH) -> (B, S, H)

        return h_state_norm

    def step(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mlstm_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        '''Single-step operation for recurrent form.

        Args:
            q (torch.Tensor): Queries tensor (B, S, H).
            k (torch.Tensor): Keys tensor (B, S, H).
            v (torch.Tensor): Values tensor (B, S, H).
            mlstm_state (tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            optional): Tensors of cell state, normalizer state and stabilizer
            state for mLSTM. Defaults to None.

        Returns:
            tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor,
            torch.Tensor]]: Output tensor and a tuple of cell state,
            normalizer state and stabilizer state for mLSTM
        '''
        B, S, _ = q.shape  # (B, S, H)

        if_gate_input = torch.cat([q, k, v], dim=-1)
        q = q.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)
        k = k.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)
        v = v.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)

        _, _, NH, DH = q.shape

        q = q.transpose(1, 2)  # (B, NH, S, DH)
        k = k.transpose(1, 2)  # (B, NH, S, DH)
        v = v.transpose(1, 2)  # (B, NH, S, DH)

        # compute input and forget gate pre-activations
        igate_preact = self.igate(if_gate_input)  # (B, S, NH)
        igate_preact = igate_preact.transpose(-1, -2).unsqueeze(
            -1
        )  # (B, NH, S, 1)
        fgate_preact = self.fgate(if_gate_input)  # (B, S, NH)
        fgate_preact = fgate_preact.transpose(-1, -2).unsqueeze(
            -1
        )  # (B, NH, S, 1)

        if mlstm_state is None:
            c_state = torch.zeros(
                size=(B, NH, DH, DH), device=q.device, dtype=q.dtype
            )
            n_state = torch.zeros(
                size=(B, NH, DH, 1), device=q.device, dtype=q.dtype
            )
            m_state = torch.zeros(
                size=(B, NH, 1, 1), device=q.device, dtype=q.dtype
            )
        else:
            c_state, n_state, m_state = mlstm_state
            c_state = c_state.to(device=q.device, dtype=q.dtype)
            n_state = n_state.to(device=q.device, dtype=q.dtype)
            m_state = m_state.to(device=q.device, dtype=q.dtype)

        h_state, mlstm_state = self.backend_fn_step(
            c_state=c_state,
            n_state=n_state,
            m_state=m_state,
            q=q,
            k=k,
            v=v,
            igate_preact=igate_preact,
            fgate_preact=fgate_preact,
        )  # (B, NH, 1 DH), ((B, NH, DH, DH), (B, NH, DH, 1), (B, NH, 1, 1))

        h_state_norm = self.outnorm(h_state)  # (B, NH, S, DH)
        h_state_norm = h_state_norm.transpose(1, 2).reshape(
            B, S, -1
        )  # (B, NH, S, DH) -> (B, S, NH, DH) -> (B, S, H)

        return h_state_norm, mlstm_state

    def reset_parameters(self) -> None:
        '''Reset parameters.'''
        self.outnorm.reset_parameters()
        # forget gate initialization
        torch.nn.init.zeros_(self.fgate.weight)
        bias_linspace_init_(self.fgate.bias, start=3.0, end=6.0)
        # input gate initialization
        torch.nn.init.zeros_(self.igate.weight)
        torch.nn.init.normal_(self.igate.bias, mean=0.0, std=0.1)


class mLSTMLayer(nn.Module):
    '''mLSTM layer.

    Inputs:
        x (torch.Tensor): Input tensor (B, S, D).
    Outputs:
        torch.Tensor: Output tensor (B, S, D)
    '''

    def __init__(
        self,
        embedding_dim: int,
        context_length: int,
        inner_embedding_dim: int = 0,
        conv1d_kernel_size: int = 4,
        qkv_proj_blocksize: int = 4,
        num_heads: int = 4,
        proj_factor: float = 2.0,
        bias: bool = False,
        dropout: float = 0.0,
        num_layers: int = 1,
        idx_layer: int = 0,
        round_proj_up_dim_up: bool = True,
        round_proj_up_to_multiple_of: int = 64,
    ) -> None:
        '''mLSTM layer.

        If the inner_embedding_dim is not specified, the inner_embedding_dim
        will be calculated according to the embedding_dim, proj_factor,
        round_proj_up_dim_up and round_proj_up_to_multiple_of.

        Args:
            embedding_dim (int, optional): Input embedding dimension.
            context_length (int, optional): Length of input for parallel form.
            inner_embedding_dim (int, optional): Inner embedding dimension of
            mLSTM layer. If not given, it will be calculated according to
            proj_factor, round_proj_up_dim_up and
            round_proj_up_to_multiple_of. Defaults to 0.
            conv1d_kernel_size (int, optional): Kernel size for 1-D causal
            convolution. Defaults to 4.
            qkv_proj_blocksize (int, optional): The block size for projecting
            for multi-head attention. Defaults to 4.
            num_heads (int, optional): Number of attention heads. Defaults to
            4.
            proj_factor (float, optional): Scaled-up factor for calculating
            inner_embedding_dim by projecting embedding input tensor to higher
            dimension. Defaults to 2.0.
            bias (bool, optional): Whether to use bias for linear layers.
            Defaults to False.
            dropout (float, optional): Drop rate for dropout. Defaults to 0.0.
            num_layers (int, optional): Number of layers in the model.
            Defaults to 1.
            idx_layer (int, optional): Index of the layer in the block.
            Defaults to 0.
            round_proj_up_dim_up (bool, optional): Whether to round the number
            of inner embedding dimension up when calculating the inner
            embedding dimension. Defaults to True.
            round_proj_up_to_multiple_of (int, optional): The base value for
            calculating the inner embedding dimension. Defaults to 64.
        '''
        super().__init__()

        self.conv1d_kernel_size = conv1d_kernel_size
        self.qkv_proj_blocksize = qkv_proj_blocksize
        self.num_heads = num_heads
        self.proj_factor = proj_factor
        self.embedding_dim = embedding_dim
        self.bias = bias
        self.context_length = context_length
        self.num_layers = num_layers
        self.idx_layer = idx_layer
        self.round_proj_up_dim_up = round_proj_up_dim_up
        self.round_proj_up_to_multiple_of = round_proj_up_to_multiple_of

        if inner_embedding_dim > 0:
            self.inner_embedding_dim = inner_embedding_dim
        else:
            if round_proj_up_dim_up:
                self.inner_embedding_dim = (
                    math.ceil(
                        proj_factor
                        * embedding_dim
                        / round_proj_up_to_multiple_of
                    )
                    * round_proj_up_to_multiple_of
                )
            else:
                self.inner_embedding_dim = (
                    math.floor(
                        proj_factor
                        * embedding_dim
                        / round_proj_up_to_multiple_of
                    )
                    * round_proj_up_to_multiple_of
                )

        self.norm = LayerNorm(ndim=embedding_dim)
        self.proj_up = nn.Linear(
            in_features=embedding_dim,
            out_features=2 * self.inner_embedding_dim,
            bias=bias,
        )

        num_proj_heads = self.inner_embedding_dim // qkv_proj_blocksize
        self.q_proj = LinearHeadwiseExpand(
            in_features=self.inner_embedding_dim,
            num_heads=num_proj_heads,
            bias=bias,
        )
        self.k_proj = LinearHeadwiseExpand(
            in_features=self.inner_embedding_dim,
            num_heads=num_proj_heads,
            bias=bias,
        )
        self.v_proj = LinearHeadwiseExpand(
            in_features=self.inner_embedding_dim,
            num_heads=num_proj_heads,
            bias=bias,
        )
        self.conv1d = CausalConv1d(
            feature_dim=self.inner_embedding_dim,
            kernel_size=conv1d_kernel_size,
        )
        self.conv_act_fn = nn.SiLU()
        self.mlstm_cell = mLSTMCell(
            context_length=context_length,
            embedding_dim=self.inner_embedding_dim,
            num_heads=num_heads,
        )
        self.ogate_act_fn = nn.SiLU()

        self.learnable_skip = nn.Parameter(
            torch.ones(self.inner_embedding_dim, requires_grad=True)
        )

        self.proj_down = nn.Linear(
            in_features=self.inner_embedding_dim,
            out_features=embedding_dim,
            bias=bias,
        )
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward method.

        Args:
            x (torch.Tensor): Input tensor (B, S, D).

        Returns:
            torch.Tensor: Output tensor (B, S, D).
        '''
        # up-projection
        x_inner = self.proj_up(self.norm(x))
        x_mlstm, z = torch.split(
            x_inner,
            split_size_or_sections=self.inner_embedding_dim,
            dim=-1,
        )

        # mlstm branch
        x_mlstm_conv = self.conv1d(x_mlstm)
        x_mlstm_conv_act = self.conv_act_fn(x_mlstm_conv)

        q = self.q_proj(x_mlstm_conv_act)
        k = self.k_proj(x_mlstm_conv_act)
        v = self.v_proj(x_mlstm)

        h_tilde_state = self.mlstm_cell(q=q, k=k, v=v)

        h_tilde_state_skip = h_tilde_state + (
            self.learnable_skip * x_mlstm_conv_act
        )

        # output / z branch
        h_state = h_tilde_state_skip * self.ogate_act_fn(z)

        # down-projection
        y = x + self.dropout(self.proj_down(h_state))

        return y

    def step(
        self,
        x: torch.Tensor,
        mlstm_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None,
        conv_state: tuple[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, tuple[torch.Tensor, ...]]]:
        '''Single-step operation for recurrent form.

        Args:
            x (torch.Tensor): Input tensor.
            mlstm_state (tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            optional): mLSTM layer states. Defaults to None.
            conv_state (tuple[torch.Tensor], optional): 1-D causal convolution
            state. Defaults to None.

        Returns:
            tuple[torch.Tensor, dict[str, tuple[torch.Tensor, ...]]]: Output
            tensor and dictionary for convolution and mLSTM layer state.
        '''
        # up-projection
        x_inner = self.proj_up(self.norm(x))
        x_mlstm, z = torch.split(
            x_inner,
            split_size_or_sections=self.inner_embedding_dim,
            dim=-1,
        )

        # mlstm branch
        x_mlstm_conv, conv_state = self.conv1d.step(
            x_mlstm, conv_state=conv_state
        )
        x_mlstm_conv_act = self.conv_act_fn(x_mlstm_conv)

        q = self.q_proj(x_mlstm_conv_act)
        k = self.k_proj(x_mlstm_conv_act)
        v = self.v_proj(x_mlstm)

        h_tilde_state, mlstm_state = self.mlstm_cell.step(
            q=q, k=k, v=v, mlstm_state=mlstm_state
        )

        h_tilde_state_skip = h_tilde_state + (
            self.learnable_skip * x_mlstm_conv_act
        )

        # output / z branch
        h_state = h_tilde_state_skip * self.ogate_act_fn(z)

        # down-projection
        y = self.proj_down(h_state)

        if self.dropout:
            y = x + self.dropout(y)

        return y, {"mlstm_state": mlstm_state, "conv_state": conv_state}

    def reset_parameters(self) -> None:
        '''Reset parameters.'''
        # init inproj
        small_init_init_(self.proj_up.weight, dim=self.embedding_dim)

        if self.proj_up.bias is not None:
            nn.init.zeros_(self.proj_up.bias)

        # init outproj
        wang_init_(
            self.proj_down.weight,
            dim=self.embedding_dim,
            num_layers=self.num_layers,
        )

        if self.proj_down.bias is not None:
            nn.init.zeros_(self.proj_down.bias)

        nn.init.ones_(self.learnable_skip)

        def _init_qkv_proj(qkv_proj: LinearHeadwiseExpand):
            # use the embedding dim instead of the inner embedding dim
            small_init_init_(qkv_proj.weight, dim=self.embedding_dim)
            if qkv_proj.bias is not None:
                nn.init.zeros_(qkv_proj.bias)

        _init_qkv_proj(self.q_proj)
        _init_qkv_proj(self.k_proj)
        _init_qkv_proj(self.v_proj)

        self.mlstm_cell.reset_parameters()


class mLSTM(nn.Module):
    '''mLSTM module with bi-directional support.

    Input:
        x (torch.Tensor): Input Tensor (B, D, S).
    Output:
        torch.Tensor: Output Tensor (B, S, C).'''

    def __init__(
        self,
        embedding_dim: int,
        num_cls: int,
        context_length: int = 1024,
        num_layers: int = 2,
        inner_embedding_dim: int = 256,
        bidirectional: bool = True,
        recurrent: bool = False,
        conv1d_kernel_size: int = 4,
        qkv_proj_blocksize: int = 4,
        num_heads: int = 4,
        proj_factor: float = 2.0,
        bias: bool = False,
        dropout: float = 0.0,
        round_proj_up_dim_up: bool = True,
        round_proj_up_to_multiple_of: int = 64,
        add_post_blocks_norm: bool = True,
    ) -> None:
        '''mLSTM module with bi-directional support.

        Args:
            embedding_dim (int): Input embedding dimension.
            context_length (int): Length of input sequences for parallel form.
            num_cls (int): Number of classes.
            num_layers (int, optional): Number of layers in the model.
            Defaults to 1.
            inner_embedding_dim (int, optional): Inner embedding dimension of
            mLSTM layer. If not given, it will be calculated according to
            proj_factor, round_proj_up_dim_up and
            round_proj_up_to_multiple_of. Defaults to 0.
            bidirectional (bool, optional): Whether to use bi-directional
            operation. Defaults to False.
            recurrent (bool, optional): Whether to run the mLSTM in recurrent
            form. Defaults to False.
            conv1d_kernel_size (int, optional): Kernel size for 1-D causal
            convolution. Defaults to 4.
            qkv_proj_blocksize (int, optional): The block size for projecting
            for multi-head attention. Defaults to 4.
            num_heads (int, optional): Number of attention heads. Defaults to
            4.
            proj_factor (float, optional): Scaled-up factor for calculating
            inner_embedding_dim by projecting embedding input tensor to higher
            dimension. Defaults to 2.0.
            bias (bool, optional): Whether to use bias for linear layers.
            Defaults to False.
            dropout (float, optional): Drop rate for dropout. Defaults to 0.25.
            round_proj_up_dim_up (bool, optional): Whether to round the number
            of inner embedding dimension up when calculating the inner
            embedding dimension. Defaults to True.
            round_proj_up_to_multiple_of (int, optional): The base value for
            calculating the inner embedding dimension. Defaults to 64.
            add_post_blocks_norm (bool, optional): Whether to add
            normalization layer at the end of the block. Defaults to True.
        '''
        super().__init__()

        self.bidirectional = bidirectional
        self.recurrent = recurrent
        self.blocks = nn.ModuleList(
            [
                mLSTMLayer(
                    embedding_dim,
                    context_length,
                    inner_embedding_dim,
                    conv1d_kernel_size,
                    qkv_proj_blocksize,
                    num_heads,
                    proj_factor,
                    bias,
                    dropout,
                    num_layers,
                    idx_layer,
                    round_proj_up_dim_up,
                    round_proj_up_to_multiple_of,
                )
                for idx_layer in range(num_layers)
            ]
        )
        self.blocks_rev = deepcopy(self.blocks) if bidirectional else None
        self.post_blocks_norm = (
            LayerNorm(embedding_dim * 2 if bidirectional else embedding_dim)
            if add_post_blocks_norm
            else nn.Identity()
        )
        self.fc = nn.Linear(
            embedding_dim * 2 if bidirectional else embedding_dim, num_cls
        )
        self.softmax = nn.Softmax(dim=2)

        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward method.

        Args:
            x (torch.Tensor): Input Tensor (B, D, S).

        Returns:
            torch.Tensor: Output Tensor (B, S, C).
        '''
        if self.recurrent:
            x = self.forward_recurrent(x)
        else:
            x = self.forward_parallel(x)

        return x

    def forward_parallel(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward function in parallel form.

        Args:
            x (torch.Tensor): Input tensor (B, D, S).

        Returns:
            torch.Tensor: Output tensor (B, S, C).
        '''
        for block in self.blocks:
            x = block(x)

        if self.bidirectional:
            x_rev = torch.flip(x, [1])

            for block_rev in self.blocks_rev:
                x_rev = block_rev(x_rev)

            x_rev = torch.flip(x, [1])
            x = torch.cat([x, x_rev], 2)

        x = self.post_blocks_norm(x)
        x = self.fc(x)
        x = self.softmax(x)

        return x

    def forward_recurrent(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward function in recurrent form.

        Args:
            x (torch.Tensor): Input tensor (B, D, S).

        Returns:
            torch.Tensor: Output tensor (B, S, C).
        '''
        x = x.permute((0, 2, 1))
        state = {}
        out = []

        for x_step in x.split(1, dim=1):
            for block_idx, block in enumerate(self.blocks):
                x_step, state[f'forward_{block_idx}'] = block.step(
                    x_step, **state.get(f'forward_{block_idx}', {})
                )

            out.append(x_step)

        x = torch.cat(out, dim=1)

        if self.bidirectional:
            x_rev = torch.flip(x, [1])
            state_rev = {}
            out_rev = []

            for x_step in x_rev.split(1, dim=1):
                for block_idx, block in enumerate(self.blocks):
                    x_step, state_rev[f'backward_{block_idx}'] = block.step(
                        x_step, **state_rev.get(f'backward_{block_idx}', {})
                    )

                out_rev.append(x_step)

            x_rev = torch.cat(out_rev, dim=1)
            x_rev = torch.flip(x_rev, [1])
            x = torch.cat([x, x_rev], 2)

        x = self.post_blocks_norm(x)
        x = self.fc(x)
        x = self.softmax(x)

        return

    def reset_parameters(self) -> None:
        '''Reset parameters.'''
        for block in self.blocks:
            block.reset_parameters()

        if not isinstance(self.post_blocks_norm, nn.Identity):
            self.post_blocks_norm.reset_parameters()
