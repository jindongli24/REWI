import argparse
import os
import warnings
from typing import Any

import torch
import torch.nn as nn
import yaml
from loguru import logger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from hwr.ctc_decoder import build_ctc_decoder
from hwr.dataset import HRDataset
from hwr.dataset.utils import fn_collate
from hwr.evaluate import evaluate
from hwr.loss import CTCLoss
from hwr.manager import RunManager
from hwr.model import BaseModel
from hwr.utils import seed_everything, seed_worker

warnings.filterwarnings('ignore', category=UserWarning)


def train_one_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    fn_loss: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    man: RunManager,
    device: str,
    epoch: int,
) -> None:
    '''Train model for 1 epoch.

    Args:
        dataloader (torch.utils.data.DataLoader): Dataloader of training set.
        model (torch.nn.Module): Model.
        fn_loss (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        scaler (torch.cuda.amp.GradScaler): Scaler for mix-precision training.
        lr_schedular (torch.optim.lr_scheduler.ReduceLROnPlateau): Learning
        rate scheduler.
        man (hwr.manager.RunManager): Running manager.
        device (str): Device to use.
        epoch (int): Current epoch number.
    '''
    man.initialize_epoch(epoch, len(dataloader), False)
    model.train()

    for idx, (x, y, len_x, len_y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        len_out = model.calculate_output_length(len_x)
        optimizer.zero_grad()
        out = model(x)
        loss = fn_loss(out.permute((1, 0, 2)), y, len_out, len_y)
        loss.backward()
        optimizer.step()
        man.update_iteration(
            idx,
            loss.item(),
            lr_scheduler.get_last_lr()[0],
        )

    man.summarize_epoch()

    # save checkpoints every freq_save epoch
    if man.check_step(epoch + 1, man.freq_save, man.epoch_max):
        man.save_checkpoint(
            model.state_dict(),
            optimizer.state_dict(),
            lr_scheduler.state_dict(),
        )


def test(
    dataloader: DataLoader,
    model: nn.Module,
    fn_loss: nn.Module,
    man: RunManager,
    ctc_decoder: Any,
    device: str,
    epoch: int = None,
) -> None:
    '''Test the model.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader of testing or
        validation set.
        model (torch.nn.Module): Model.
        fn_loss (torch.nn.Module): Loss function.
        man (hwr.manager.RunManager): Running manager.
        decoder (typing.Any): An instance of CTC decoder.
        categories (list[str]): Category infomation for evaluation.
        device (str): Device to use.
        epoch (int, optional): Epoch number. Defaults to None.
    '''
    preds = []  # predictions for evaluation
    labels = []  # labels for evaluation
    man.initialize_epoch(epoch, len(dataloader), True)
    model.eval()

    with torch.no_grad():
        for idx, (x, y, len_x, len_y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            len_out = model.calculate_output_length(len_x)
            out = model(x)
            loss = fn_loss(out.permute((1, 0, 2)), y, len_out, len_y)
            man.update_iteration(idx, loss.item())

            # decode and cache results every freq_eval epoch
            if man.check_step(epoch + 1, man.freq_eval, man.epoch_max):
                for pred, len_pred, label in zip(out.cpu(), len_out, y.cpu()):
                    preds.append(ctc_decoder.decode(pred[:len_pred]))
                    labels.append(ctc_decoder.decode(label, True))

    loss = man.summarize_epoch()

    # evaluate every freq_eval epoch
    if man.check_step(epoch + 1, man.freq_eval, man.epoch_max):
        results_eval = evaluate(preds=preds, labels=labels)
        man.update_evaluation(results_eval, preds[:20], labels[:20])

def main(cfgs: argparse.Namespace) -> None:
    '''Training or evaluate.

    Args:
        cfgs (argparse.Namespace): Configurations.
    '''
    # initialize the environment
    manager = RunManager(cfgs)
    seed_everything(cfgs.seed)
    ctc_decoder = build_ctc_decoder(cfgs.categories, cfgs.ctc_decoder)

    # initialize the datasets and dataloaders
    dataset_test = HRDataset(
        path_anno=os.path.join(
            cfgs.dir_dataset, 'val.json'
        ),
        categories=cfgs.categories,
        sensors=cfgs.sensors,
        ratio_ds=8,
        idx_cv=cfgs.idx_cv,
        cache=cfgs.cache,
    )
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=cfgs.size_batch,
        num_workers=cfgs.num_worker,
        collate_fn=fn_collate,
    )
    fn_loss = CTCLoss()
    model = BaseModel(
        cfgs.arch_en,
        cfgs.arch_de,
        cfgs.in_chan,
        cfgs.num_cls,
        8,
        0,
    ).to(cfgs.device)
    epoch_start = 0

    if not cfgs.test:
        dataset_train = HRDataset(
            path_anno=os.path.join(cfgs.dir_dataset, 'train.json'),
            categories=cfgs.categories,
            sensors=cfgs.sensors,
            ratio_ds=8,
            idx_cv=cfgs.idx_cv,
            cache=cfgs.cache,
        )
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=cfgs.size_batch,
            shuffle=True,
            num_workers=cfgs.num_worker,
            collate_fn=fn_collate,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(cfgs.seed),
        )
        optimizer = torch.optim.Adam(model.parameters(), cfgs.lr)
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.8, min_lr=1e-4)

    # load checkpoint if given
    if cfgs.checkpoint:
        ckp = torch.load(cfgs.checkpoint, weights_only=False)
        model.load_state_dict(ckp['model'])

        if not cfgs.test:
            epoch_start = ckp['epoch'] + 1
            optimizer.load_state_dict(ckp['optimizer'])
            lr_scheduler.load_state_dict(ckp['lr_scheduler'])

        logger.info(f'Load checkpoint from {cfgs.checkpoint}')

    # start running
    losses_val = []

    for e in range(epoch_start, 1000):
        if cfgs.test:
            test(
                dataloader=dataloader_test,
                model=model,
                fn_loss=fn_loss,
                man=manager,
                ctc_decoder=ctc_decoder,
                device=cfgs.device,
                epoch=-1,
            )
            break
        else:
            train_one_epoch(
                dataloader=dataloader_train,
                model=model,
                fn_loss=fn_loss,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                man=manager,
                device=cfgs.device,
                epoch=e,
            )
            loss_val = test(
                dataloader=dataloader_test,
                model=model,
                fn_loss=fn_loss,
                lr_scheduler=lr_scheduler,
                man=manager,
                ctc_decoder=ctc_decoder,
                device=cfgs.device,
                epoch=e,
            )

            if lr_scheduler.get_last_lr()[0] <= 1e-4:
                if len(losses_val) >= 20:
                    if loss_val > losses_val[-20]:
                        break

                losses_val.append(loss_val)

    if not cfgs.test:
        manager.summarize_evaluation()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run CTC for handwriting recognition.'
    )
    parser.add_argument(
        '-c', '--config', help='Path to YAML file of configuration.'
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfgs = yaml.safe_load(f)
        cfgs = argparse.Namespace(**cfgs)

    main(cfgs)
