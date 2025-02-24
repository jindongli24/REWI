from typing import Any

import torch


class BestPath:
    '''Greedy CTC decoder for converting model outputs to sequences. The
    seperator ('') is denoted as 0 and symbols are denoted as an additional
    category (len(categiries) + 1).
    '''

    def __init__(self, categories: list[str]) -> None:
        '''Greedy CTC decoder for converting model outputs to sequences.

        Args:
            categories (list[str]): String of categories. Every character in
            the string is a individual class.
        '''
        self.categories = categories

    def decode(self, seq: torch.Tensor, label: bool = False) -> str:
        '''Decode the input sequence. If the input sequence is not a label,
        the consecutive repetative values will be removed.

        Args:
            sequence (torch.Tensor): Outputs of CTC models or labels. Shape:
            [length_sequence, number_categories] (for prediction) or
            [length_sequence] (for label)
            label (bool, optional): Whether the input sequence is a label
            sequence. Defaults to False.

        Returns:
            str: Predicted sentences.
        '''
        if not label:
            seq = torch.argmax(seq, dim=-1)
            seq = torch.unique_consecutive(seq, dim=-1)

        seq = [self.categories[i] for i in seq]
        seq = ''.join(seq)

        return seq


def build_ctc_decoder(categories: list[str], type: str) -> Any:
    '''Build CTC decoder.

    Args:
        categories (list[str]): List of categories including the seperator.
        type (str): Name of the CTC decoder to use. Option is "best_path".

    Returns:
        Any: CTC decoder.
    '''
    # TODO: Beam search CTC decoder
    match type:
        case 'best_path':
            return BestPath(categories)
