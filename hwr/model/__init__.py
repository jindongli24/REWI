import torch
import torch.nn as nn

from .ablation import AblaDec, AblaEnc
from .conv import BLCNN
from .conv_v2 import BLCNNv2
from .lstm import LSTM
from .others.convnext import ConvNeXt
from .others.mlp_mixer import MLPMixer
from .others.mlstm import mLSTM
from .others.swin import SwinTransformerV2
from .others.transformer import TransDec, TransEnc
from .others.resnet import ResNet
from .previous.felix import FelBiLSTM, FelCNN
from .previous.mohamad import MohDec, MohEnc


def build_encoder(in_chan: int, arch: str, len_seq: int = 0) -> nn.Sequential:
    '''Build encoder for CTC model.

    Args:
        in_chan (int): Number of input channels.
        arch (str, optional): Encoder architecture.
        len_seq (int, optional): Length of input sequence. Defaults to 0.

    Returns:
        nn.Sequential: Encoder.
    '''
    match arch:
        case 'mohamad':
            return MohEnc(in_chan)
        case 'felix_cnn':
            return FelCNN(in_chan)
        case 'resnet':
            return ResNet(in_chan)
        case 'convnext':
            return ConvNeXt(in_chan)
        case 'mlp_mixer':
            return MLPMixer(in_chan, len_seq)
        case 'trans':
            return TransEnc(in_chan, len_seq)
        case 'swinv2':
            return SwinTransformerV2(in_chan, len_seq)
        case 'abla':
            return AblaEnc(in_chan, True, True, True, True, True, True, True)
        case 'blcnn':
            return BLCNN(in_chan)
        case 'blcnnv2':
            return BLCNNv2(in_chan)


def build_decoder(
    size_in: int, num_cls: int, arch: str, len_seq: int = 0
) -> nn.Sequential | nn.Module:
    '''Build decoder for CTC model

    Args:
        size_in (int): Size of input.
        num_cls (int): Number of categories.
        arch (str): Architecture to use.
        len_seq (int, optional): Length of input sequence. Defaults to 0.

    Returns:
        nn.Sequential | nn.Module: Decoder.
    '''
    match arch:
        case 'mohamad':
            return MohDec(size_in, num_cls)
        case 'felix':
            return FelBiLSTM(size_in, num_cls)
        case 'trans':
            return TransDec(size_in, num_cls)
        case 'mlstm':
            return mLSTM(size_in, num_cls, len_seq)
        case 'abla':
            return AblaDec(size_in, num_cls, True, True, True, True)
        case 'lstm':
            return LSTM(size_in, num_cls)


class BaseModel(nn.Module):
    def __init__(
        self,
        arch_en: str,
        arch_de: str,
        in_chan: int,
        num_cls: int,
        ratio_ds: int,
        len_seq: int = 0,
    ) -> None:
        '''CLDNN model for handwriting recognition with CTC (Connectionist
        Temporal Classification).

        Args:
            in_chan (int): Number of input channels.
            num_cls (int): Number of classes.
        '''
        super().__init__()

        self.arch_en = arch_en
        self.arch_de = arch_de
        self.in_chan = in_chan
        self.num_cls = num_cls
        self.len_seq = len_seq
        self.ratio_ds = ratio_ds

        self.encoder = build_encoder(in_chan, arch_en, len_seq)
        self.decoder = build_decoder(
            self.encoder.size_out, num_cls, arch_de, len_seq // ratio_ds
        )

    def calculate_output_length(self, length: torch.Tensor) -> torch.Tensor:
        '''Calculate the output length of padded inputs.

        Args:
            length (torch.Tensor): Lengths of padded input.

        Returns:
            torch.Tensor: Processed lengths.
        '''
        return length // self.ratio_ds

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward method.

        Args:
            x (torch.Tensor): Input tensor with a shape of (size_batch,
            num_chan, len_seq).

        Returns:
            torch.Tensor: Output tensor with a shape of (size_batch,
            len_seq // ratio_ds, num_cls).
        '''
        x = self.encoder(x)
        x = self.decoder(x)

        return x

    def fuse(self) -> None:
        '''Fuse parameters of layers.'''
        if hasattr(self.encoder, 'fuse'):
            self.encoder.fuse()

        if hasattr(self.decoder, 'fuse'):
            self.decoder.fuse()
