import jiwer
import Levenshtein
import numpy as np


def get_levenshtein_distance(
    preds: list[str], labels: list[str]
) -> tuple[float, float]:
    '''Calculate the Levenshtein distance between predictions and labels.

    Args:
        preds (list[str]): Predictions.
        labels (list[str]): Ground-truth labels.

    Returns:
        tuple[float, float]: Levenshtein distance and average length of labels.
    '''
    dist_leven = []
    len_label_avg = []

    for pred, label in zip(preds, labels):
        dist = Levenshtein.distance(pred, label)
        dist_leven.append(dist)
        len_label_avg.append(len(label))

    dist_leven = np.mean(dist_leven)
    len_label_avg = np.mean(len_label_avg)

    return dist_leven, len_label_avg


def evaluate(
    preds: str | list[str],
    labels: str | list[str],
    use_ld: bool = True,
    use_cer: bool = True,
    use_wer: bool = True,
) -> dict:
    '''Prediction evaluation.

    Args:
        preds (str | list[str]): Sentences of predictions.
        labels (str | list[str]): Sentences of ground-truth labels.
        use_ld (bool, optional): Whether to calculate the Levenshtein distance and average length of ground-truth sentences. Defaults to True.
        use_cer (bool, optional): Whether to calculate the character error rate. Defaults to True.
        use_wer (bool, optional): Whether to calculate the word error rate. Defaults to True.

    Returns:
        dict: Results of Levenshtein distance, average sentence length, character error rate and word error rate. The values that are not calculated are shown as -1.
    '''
    if isinstance(preds, str):
        preds = [preds]

    if isinstance(labels, str):
        labels = [labels]

    if use_ld:
        dist_leven, len_sent_avg = get_levenshtein_distance(preds, labels)
    else:
        dist_leven, len_sent_avg = -1, -1

    cer = jiwer.cer(labels, preds) if use_cer else -1
    wer = jiwer.wer(labels, preds) if use_wer else -1

    return {
        'levenshtein_distance': dist_leven,
        'average_sentence_length': len_sent_avg,
        'character_error_rate': cer,
        'word_error_rate': wer,
    }
