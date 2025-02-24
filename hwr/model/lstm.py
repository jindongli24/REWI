import torch
import torch.nn as nn

__all__ = ['LSTM']


class LSTM(nn.Module):
    '''Bi-LSTM module for classification.

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
        hidden_size: int = 128,
        num_layers: int = 3,
        r_drop: float = 0.2,
    ) -> None:
        '''Bi-LSTM module for classification.

        Args:
            size_in (int): Number of input channel.
            num_cls (int): Number of categories.
            hidden_size (int): Hidden size of LSTM. Defaults to 128.
            num_layers (int, optional): Number of LSTM layers. Defaults to 3.
            r_drop (float): Dropping rate for dropout layers. Defaults to 0.2.
        '''
        super().__init__()

        self.lstm = nn.LSTM(
            size_in,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=r_drop,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, num_cls)
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
        x = self.fc(x)
        x = self.softmax(x)

        return x
