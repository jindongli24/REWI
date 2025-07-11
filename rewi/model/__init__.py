import torch
import torch.nn as nn

from .ablation import AblaDec, AblaEnc
from .conv import BLConv
from .lstm import LSTM
from .others.convnext import ConvNeXt
from .others.mlp_mixer import MLPMixer
from .others.resnet import ResNet
from .others.swin import SwinTransformerV2
from .others.vit import ViT
from .previous.cldnn import CLDNNDec, CLDNNEnc
from .previous.ott import OttBiLSTM, OttCNN


def build_encoder(in_chan: int, arch: str, len_seq: int = 0) -> nn.Module:
    '''Build encoder for CTC model.

    Args:
        in_chan (int): Number of input channels.
        arch (str, optional): Encoder architecture.
        len_seq (int, optional): Length of the input sequence. Defaults to 0.

    Returns:
        torch.nn.Module: Encoder.
    '''
    match arch:
        case 'blconv_b':
            return BLConv(in_chan)
        case 'blconv_s':
            return BLConv(in_chan, [1, 1, 1], [64, 128, 256])
        case 'cldnn':
            return CLDNNEnc(in_chan)
        case 'ott':
            return OttCNN(in_chan)
        case 'convnext':
            return ConvNeXt(in_chan)
        case 'mlp_mixer':
            return MLPMixer(in_chan, len_seq)
        case 'resnet':
            return ResNet(in_chan)
        case 'swinv2':
            return SwinTransformerV2(in_chan, len_seq)
        case 'vit':
            return ViT(in_chan, len_seq)
        case 'abla':
            return AblaEnc(
                in_chan,
                True,
                True,
                True,
                True,
                True,
                True,
            )


def build_decoder(
    dim_in: int, num_cls: int, arch: str, len_seq: int = 0
) -> nn.Module:
    '''Build decoder for CTC model

    Args:
        dim_in (int): Number of input dimensions.
        num_cls (int): Number of categories.
        arch (str): Architecture to use.
        len_seq (int, optional): Length of the input sequence. Defaults to 0.

    Returns:
        torch.nn.Module: Decoder.
    '''
    match arch:
        case 'bilstm_b':
            return LSTM(dim_in, num_cls)
        case 'bilstm_s':
            return LSTM(dim_in, num_cls, 64, 2)
        case 'cldnn':
            return CLDNNDec(dim_in, num_cls)
        case 'ott':
            return OttBiLSTM(dim_in, num_cls)
        case 'abla':
            return AblaDec(dim_in, num_cls)


class BaseModel(nn.Module):
    '''Handwriting recognition model using CTC (Connectionist Temporal
    Classification).

    Inputs:
        x (torch.Tensor): Input tensor with a shape of (size_batch, num_chan, len_seq).
    Outputs:
        torch.Tensor: Output tensor with a shape of (size_batch, len_seq // ratio_ds, num_cls).
    '''

    def __init__(
        self,
        arch_en: str,
        arch_de: str,
        in_chan: int,
        num_cls: int,
        len_seq: int = 0,
    ) -> None:
        '''Handwriting recognition model using CTC (Connectionist Temporal
        Classification).

        Args:
            arch_en (str): Name of the encoder.
            arch_de (str): Name of the decoder.
            in_chan (int): Number of input channels.
            num_cls (int): Number of classes.
            len_seq (int, optional): Length of the input sequence. Defaults to 0.
        '''
        super().__init__()

        self.arch_en = arch_en
        self.arch_de = arch_de
        self.in_chan = in_chan
        self.num_cls = num_cls
        self.len_seq = len_seq

        self.encoder = build_encoder(in_chan, arch_en, len_seq)
        self.decoder = build_decoder(
            self.encoder.dim_out,
            num_cls,
            arch_de,
            len_seq // self.encoder.ratio_ds if arch_en != 'trans' else 0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward method.

        Args:
            x (torch.Tensor): Input tensor with a shape of (size_batch, num_chan, len_seq).

        Returns:
            torch.Tensor: Output tensor with a shape of (size_batch, len_seq // ratio_ds, num_cls).
        '''
        x = self.encoder(x)
        x = self.decoder(x)

        return x

    def infer(self) -> None:
        '''Switch the model to inference mode.'''
        # blconv: fuse parameters of layers
        if hasattr(self.encoder, 'fuse'):
            self.encoder.fuse()

        # bimlstm: switch to recurrent mode
        if hasattr(self.decoder, 'recurrent'):
            self.decoder.recurrent = True

    @property
    def ratio_ds(self) -> int:
        '''Get the downsample ratio between input length and output length.

        Returns:
            int: Downsample ratio.
        '''
        return self.encoder.ratio_ds
