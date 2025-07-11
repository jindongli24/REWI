import torch
from torch.nn.utils.rnn import pad_sequence


def fn_collate(batch: list[tuple[torch.Tensor]]) -> tuple[torch.Tensor]:
    '''Collate function for aligning the shape of data sequences and labels.

    Args:
        batch (list[tuple[torch.Tensor]]): Input batch data.

    Returns:
        tuple[torch.Tensor]: Aligned batch data of sequences and labels.
    '''
    seqs, labels, lens_sig, lens_label = [], [], [], []

    for x, y in batch:
        seqs.append(x)
        labels.append(y)
        lens_sig.append(len(x))
        lens_label.append(len(y))

    seqs = pad_sequence(seqs, True).permute(0, 2, 1)
    labels = pad_sequence(labels, True)
    lens_sig = torch.tensor(lens_sig)
    lens_label = torch.tensor(lens_label)

    return seqs, labels, lens_sig, lens_label
