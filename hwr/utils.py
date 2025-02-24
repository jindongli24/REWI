import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    '''Seed everything in the environment.

    Args:
        seed (int, optional): Seed value. Defaults to 42.
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    '''Seed workers of dataloader.

    Args:
        worker_id (int): Worker ID.
    '''
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def sec2time(time_sec: float) -> str:
    '''Convert number of seconds to time in h:m:s format.

    Args:
        time (float): Number of seconds.

    Returns:
        str: String of time.
    '''
    second = str(int(time_sec % 60)).zfill(2)
    minute = str(int(time_sec // 60) % 60).zfill(2)
    hour = int(time_sec // 3600)
    time = f'{hour}:{minute}:{second}'

    return time
