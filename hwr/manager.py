import argparse
import json
import os
import time
from datetime import datetime
from typing import Any

import torch
import yaml
from loguru import logger

from .utils import sec2time


class RunManager:
    '''Manage work directory, logging, losses, evaluation results and
    checkpoints.
    '''

    def __init__(self, cfgs: argparse.Namespace) -> None:
        '''Manage work directory, logging, losses, evaluation results and
        checkpoints.

        Args:
            cfgs (argparse.Namespace): Configurations.
        '''
        self.dir_work = cfgs.dir_work
        self.epoch_max = cfgs.epoch
        self.freq_eval = cfgs.freq_eval
        self.freq_log = cfgs.freq_log
        self.freq_save = cfgs.freq_save
        self.test = cfgs.test

        self.results = {}
        self.ts = datetime.now().strftime('%Y%m%d%H%M%S')

        self.initialize_directory(cfgs)

    def check_step(self, scur: int, freq: int, smax: int) -> bool:
        '''Check whether the current step is desired according to the given
        frequency. The last step is always True regardless the frequency.

        Args:
            scur (int): Current step number.
            freq (int): Frequency.
            smax (int): Maximum step.

        Returns:
            bool: Whether the current step is desired.
        '''
        if scur % freq == 0 or scur == smax:
            return True
        else:
            return False

    def initialize_directory(self, cfgs: argparse.Namespace) -> None:
        '''Initialize the work directory.

        Args:
            cfgs (argparse.Namespace): Configurations.
        '''
        tag = 'test' if self.test else 'train'
        idx_cv = str(cfgs.idx_cv)

        self.dir_ckp = os.path.join(cfgs.dir_work, idx_cv, 'checkpoints')
        path_cfg = os.path.join(cfgs.dir_work, idx_cv, f'{tag}_{self.ts}.yaml')
        path_log = os.path.join(cfgs.dir_work, idx_cv, f'{tag}_{self.ts}.log')
        self.path_result = os.path.join(
            cfgs.dir_work, idx_cv, f'{tag}_{self.ts}.json'
        )

        if not os.path.isdir(cfgs.dir_work):
            os.mkdir(cfgs.dir_work)

        if not os.path.isdir(os.path.join(cfgs.dir_work, idx_cv)):
            os.mkdir(os.path.join(cfgs.dir_work, idx_cv))

        if not os.path.isdir(self.dir_ckp):
            os.mkdir(self.dir_ckp)

        with open(os.path.join(path_cfg), 'w') as f:
            yaml.safe_dump(vars(cfgs), f)

        logger.add(path_log)
        logger.info(
            f'Initialized work directory at {cfgs.dir_work} for fold {idx_cv}.'
        )

    def initialize_epoch(self, epoch: int, num_iter: int, val: bool) -> None:
        '''Initialize the recording for new epoch.

        Args:
            epoch (int): Epoch number.
            num_iter (int): Maximum number of iterations of the current epoch.
            val (bool): Whether the current epoch is for training phase, or
            validation or test phases.
        '''
        self.epoch = epoch
        self.loss = []
        self.num_iter = num_iter
        self.t_start = time.time()
        self.tag = 'test' if val else 'train'

        if not epoch in self.results.keys():
            self.results[epoch] = {}

        self.results[epoch][self.tag] = []

    def log(self, message: str) -> None:
        '''Log messages.

        Args:
            message (str): Message to log.
        '''
        logger.info(message)

    def save_checkpoint(
        self,
        state_model: dict = None,
        state_optimizer: dict = None,
        state_lr_scheduler: dict = None,
    ) -> None:
        '''Save the states of model, optimizer and scheduler to the checkpoint
        directory in the work directory.

        Args:
            state_model (dict, optional): State dictionary of the model.
            Defaults to None.
            state_optimizer (dict, optional): State dictionary of the
            optimizer. Defaults to None.
            state_lr_scheduler (dict, optional): State dictionary of the
            learning rate scheduler. Defaults to None.
        '''
        torch.save(
            {
                'epoch': self.epoch,
                'lr_scheduler': state_lr_scheduler,
                'model': state_model,
                'optimizer': state_optimizer,
            },
            os.path.join(self.dir_ckp, f'{self.epoch}.pth'),
        )
        logger.info(f'Saved checkpoint of epoch {self.epoch}')

    def save_results(self) -> None:
        '''Save the cached result dictionary as a JSON file to the work
        directory.'''
        with open(self.path_result, 'w') as f:
            json.dump(self.results, f)

    def summarize_evaluation(self) -> None:
        '''Summarize the evaluation results.'''
        results_eval = [
            [epoch, result['evaluation']]
            for epoch, result in self.results.items()
            if 'evaluation' in result.keys()
        ]
        metrics = results_eval[0][1].keys()
        best = {metric: [-1, -1] for metric in metrics}  # [epoch, value]

        for result in results_eval:
            for metric in metrics:
                if (
                    result[1][metric] < best[metric][1]
                    or best[metric][0] == -1
                ):
                    best[metric] = [result[0], float(result[1][metric])]

        self.results['best'] = best
        logger.info(f'best: {best}')
        self.save_results()

    def summarize_epoch(self) -> float:
        '''Summarize and save the results of the epoch.

        Returns:
            float: Average loss of the epoch.
        '''
        t_end = time.time() - self.t_start
        loss_avg = sum(self.loss) / len(self.loss)
        result = {'loss_avg': loss_avg, 'time': t_end}
        logger.info(
            (
                f'{self.tag}, epoch: {self.epoch}, loss avg: {loss_avg:.7f}, '
                f'time: {sec2time(t_end)}'
            )
        )
        self.results[self.epoch][self.tag].append(result)
        self.save_results()

        return loss_avg

    def update_evaluation(
        self, result: dict, preds: Any = None, labels: Any = None
    ) -> None:
        '''Update the evaluation results. Log predictions and labels if they
        are given.

        Args:
            result (dict): Evaluation results.
            preds (Any, optional): Predictions. Defaults to None.
            labels (Any, optional): Labels. Defaults to None.
        '''
        self.results[self.epoch]['evaluation'] = result
        self.save_results()
        msg_log = [f'{key}: {val:.7f} ' for key, val in result.items()]
        logger.info(', '.join(msg_log))

        if preds:
            logger.info(f'predictions: {preds}')

        if labels:
            logger.info(f'labels: {labels}')

    def update_iteration(
        self,
        iter: int,
        loss: float,
        lr: float = -1,
    ) -> None:
        '''Update the status of the iteration. If the current iteration is
        desired according to the given frequency, log the information.

        Args:
            iter (int): Iteration number.
            loss (float): Loss value.
            lr (float, optional): Current learning rate. Defaults to -1.
        '''
        self.loss.append(loss)

        if self.check_step(iter + 1, self.freq_log, self.num_iter):
            t_inter = time.time() - self.t_start
            result = {
                'lr': lr,
                'iters': iter + 1,
                'loss': loss,
                'time': t_inter,
            }
            self.results[self.epoch][self.tag].append(result)
            logger.info(
                (
                    f'{self.tag}, epoch: {self.epoch}, iters: {iter + 1}/'
                    f'{self.num_iter}, lr: {lr:.7f}, loss: {loss:.7f}, time: '
                    f'{sec2time(t_inter)}'
                )
            )
