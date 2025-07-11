import os

import jiwer
import matplotlib.pyplot as plt
import numpy as np


def draw_sem(
    refs: list[str],
    hyps: list[str],
    cats: list[str],
    path_save: str = 'mat_se.pdf',
) -> None:
    '''Draw substitution error matrix. Substitution error matrix is a variant
    of confusion matrix that only consider the substitution error in text
    sequences. For better visualization, the color of the gray matrix is
    reversed, i.e. black is 1 and white is 0.

    Args:
        refs (list[str]): Labels.
        hyps (list[str]): Predictions.
        cats (list[str]): Categories.
        dir_save (str, optional): Path to the PDF image file to save the figure. Defaults to 'mat_se.pdf'.
    '''
    out = jiwer.process_characters(refs, hyps)
    confusion = np.zeros((len(cats), len(cats)))
    count = np.zeros((1, len(cats)))

    cnt_event = {'delete': 0, 'equal': 0, 'insert': 0, 'substitute': 0}

    for results, hyp, ref in zip(
        out.alignments, out.hypotheses, out.references
    ):
        for event in results:
            if event.type in ['substitute']:
                for i in range(event.ref_start_idx, event.ref_end_idx):
                    for j in range(event.hyp_start_idx, event.hyp_end_idx):
                        confusion[cats.index(hyp[j])][cats.index(ref[i])] += 1

            cnt_event[event.type] += (
                event.hyp_end_idx - event.hyp_start_idx
                if event.type != 'delete'
                else event.ref_end_idx - event.ref_start_idx
            )

        for char in ref:
            count[0][cats.index(char)] += 1

    plt.figure(figsize=(10, 10), dpi=300)
    plt.figtext(
        0.5,
        0.01,
        ''.join([f'{k}: {v}, ' for k, v in cnt_event.items()]),
        ha='center',
    )
    plt.imshow(1 - (confusion / count), cmap='gray')
    plt.xlabel('Reference')
    plt.ylabel('Hypothesis')
    plt.xticks(np.arange(len(cats)), cats)
    plt.yticks(np.arange(len(cats)), cats)
    plt.title('Substitution Error Matrix')
    plt.tight_layout(rect=[0.0, 0.02, 1, 0.98])
    plt.savefig(path_save)
    plt.close()


def visualize(
    preds: str | list[str],
    labels: str | list[str],
    cats: list[str],
    dir_save: str,
    epoch: int,
    use_sem: bool = True,
) -> None:
    '''Results visualization.

    Args:
        preds (str | list[str]): Sentences of predictions.
        labels (str | list[str]): Sentences of ground-truth labels.
        cats (list[str]): Categories.
        dir_save (str): Path to the directory to save.
        epoch (int): Epoch number.
        use_sem (bool, optional): Whether to draw the substitution error matrix. Defaults to True.
    '''    
    if use_sem:
        path_save = os.path.join(dir_save, f'mat_se_{epoch}.pdf')
        draw_sem(labels, preds, cats, path_save)
