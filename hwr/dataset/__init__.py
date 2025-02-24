import json
import os

import numpy as np
import torch
from loguru import logger
from torch.nn.functional import pad
from torch.utils.data import Dataset
from tqdm import tqdm

from .transforms import AddNoise, Drift, Dropout, TimeWarp

SENSOR = {
    'AF': [0, 1, 2], # front accelerometer
    'AR': [3, 4, 5], # rear accelerometer
    'G': [6, 7, 8], # gyroscope
    'M': [9, 10, 11], # magnetometer
    'F': [12], # force sensor
}


class HRDataset(Dataset):
    '''Dataset for handwriting recognition.'''

    def __init__(
        self,
        path_anno: str,
        categories: list[str],
        sensors: list[str],
        ratio_ds: int,
        idx_cv: str | int,
        size_window: int = 1,
        aug: bool = False,
        len_seq: int = 0,
        cache: bool = False,
    ) -> None:
        '''Dataset for handwriting recognition.

        Args:
            path_anno (str): Path to the annotation file of the dataset.
            categories (list[str]): List of categories.
            sensors (list[str]): Sensors to use.
            ratio_ds (int): Downsampling ratio of the model.
            idx_cv (str | int): Fold index for cross validation.
            size_window (int, optional): Windows size for window-based
            attention model. Defaults to 1.
            aug (bool): Whether to augment data. Defaults to False.
            len_seq (int, optional): Length of sequence to pad. Defaults to 0.
            cache (bool, optional): Whether to cache the data to speed up the
            data processing. Defaults to False.
        '''
        self.dir_ds = os.path.dirname(path_anno)
        self.categories = categories
        self.ratio_ds = ratio_ds
        self.idx_cv = idx_cv
        self.size_window = size_window
        self.len_seq = len_seq
        self.cache = cache
        self.idx_channel = []

        self.augs = (
            [
                AddNoise(scale=0.05, kind='multiplicative'),
                Drift(0.1, 40, 'multiplicative'),
                Dropout(size=(5, 10), per_channel=True),
                TimeWarp(5, 4),
            ]
            if aug
            else None
        )

        for sensor in sensors:
            self.idx_channel.extend(SENSOR[sensor])

        with open(path_anno, 'r') as f:
            annos = json.load(f)
            self.annos = annos['annotations'][str(idx_cv)]

        if cache:
            self.cache = cache
            self.data_cache = [
                [
                    np.loadtxt(
                        os.path.join(self.dir_ds, anno['filename']),
                        delimiter=';',
                        dtype=np.float32,
                    ),
                    anno['label'],
                ]
                for anno in tqdm(self.annos)
            ]
            logger.info(f'Cached dataset {path_anno}')

    def __len__(self) -> int:
        '''Get number of data sequences in the dataset

        Returns:
            int: Number of data sequences.
        '''

        return len(self.annos)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        '''Get item according to index number.

        Args:
            index (int): Index of data sequence.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple of data sequence and label.
        '''
        # load data
        if self.cache:
            seq, label = self.data_cache[idx]
        else:
            anno = self.annos[idx]
            seq = np.loadtxt(
                os.path.join(self.dir_ds, anno['filename']),
                delimiter=';',
                dtype=np.float32,
            )
            label = anno['label']

        # label pre-processing
        label = [self.categories.index(char) for char in label]  # ctc encode
        label = torch.tensor(label, dtype=torch.int32)

        # sequence pre-processing
        seq = seq[:, self.idx_channel]
        seq = self.process(seq, len(label))

        return seq, label

    def process(self, seq: np.ndarray, len_label: int) -> torch.Tensor:
        '''Pre processing.

        Args:
            seq (numpy.ndarray): Sequence to process.
            len_label (int): Length of the label.

        Returns:
            torch.Tensor: Processed sequence.
        '''
        # data augmentation
        if self.augs is not None:
            for aug in self.augs:
                if np.random.random() < 0.25:
                    seq = aug(seq)

        # normalize
        seq = (seq - np.mean(seq, 0)) / (np.std(seq, 0) + 1e-6)
        seq = torch.from_numpy(seq).to(torch.float32)

        # padding
        if self.len_seq > 0:
            seq = pad(seq.T, (0, self.len_seq - len(seq))).T
        else:
            if len(seq) < (len_min := len_label * 2 * self.ratio_ds):
                seq = pad(seq.T, (0, len_min - len(seq))).T

            if remain := (len(seq) % (self.ratio_ds * self.size_window)):
                seq = pad(
                    seq.T, (0, self.ratio_ds * self.size_window - remain)
                ).T

        return seq
