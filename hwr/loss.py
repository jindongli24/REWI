import torch
import torch.nn as nn


def smooth_probs(probs: torch.Tensor, alpha: float = 1e-6) -> torch.Tensor:
    '''Smooth a probability distribution with a given smoothing factor for 
    stable convergence.

    Args:
        probs (torch.Tensor): Original probability distribution of shape
        (batch_size, len_seq, num_cls).
        alpha (float, optional): Smoothing factor. Defaults to 1e-6.

    Returns:
        torch.Tensor: Smoothed probability distribution of the same shape as
        input.
    '''
    num_cls = probs.shape[-1]
    distr_uni = torch.full_like(probs, 1.0 / num_cls)
    probs = (1 - alpha) * probs + alpha * distr_uni
    # ensure the smoothed probabilities sum to 1 along the last dimension
    probs /= probs.sum(dim=-1, keepdim=True)

    return probs


class CTCLoss(nn.Module):
    def __init__(
        self,
        alpha_smooth: float = 1e-6,
        blank: int = 0,
        reduction: str = 'mean',
        zero_infinity: bool = False,
    ) -> None:
        '''Custom CTCLoss with probability smoothing.

        Args:
            alpha_smooth (float, optional): Smooth factor for input
            probability smoothing. If the factor is 0, original probability
            predictions are used. Defaults to 1e-6.
            blank (int, optional): Blank label. Defaults to 0.
            reduction (str, optional): Specifies the reduction to apply to the
            output. Options are 'none', 'mean' and 'sum'. Defaults to 'mean'.
            zero_infinity (bool, optional): Whether to zero infinite losses
            and the associated gradients. Defaults to False.
        '''
        super().__init__()

        self.alpha_smooth = alpha_smooth
        self.blank = blank
        self.reduction = reduction
        self.zero_infinity = zero_infinity

    def forward(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        '''Forward method.

        Args:
            probs (torch.Tensor): Non-log probability predictions. (T, N, C)
            or (T, C) where C = number of characters in alphabet including
            blank, T = input length, and N = batch size.
            targets (torch.Tensor): Targets. (N, S) or (sum(target_lengths)).
            input_lengths (torch.Tensor): (N) or (). Lengths of the inputs
            (must each be <= T)
            target_lengths (torch.Tensor): (N) or (). Lengths of the targets.

        Returns:
            torch.Tensor: Loss tensor.
        '''
        if self.alpha_smooth:
            probs = smooth_probs(probs, self.alpha_smooth)

        probs = probs.log()
        loss = nn.functional.ctc_loss(
            probs,
            targets,
            input_lengths,
            target_lengths,
            self.blank,
            self.reduction,
            self.zero_infinity,
        )

        return loss
