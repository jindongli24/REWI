import argparse
import os
import warnings

import torch
import torch.nn as nn
import yaml
from loguru import logger
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from rewi.dataset import HRDataset
from rewi.dataset.utils import fn_collate
from rewi.evaluate import evaluate
from rewi.loss import CTCLoss
from rewi.manager import RunManager
from rewi.model import BaseModel
from rewi.utils import seed_everything, seed_worker
from rewi.visualize import visualize

warnings.filterwarnings('ignore', category=UserWarning)


def train_one_epoch(
    dataloader: DataLoader,
    model: BaseModel,
    fn_loss: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    lr_scheduler: torch.optim.lr_scheduler.SequentialLR,
    man: RunManager,
    epoch: int,
) -> None:
    '''Train model for 1 epoch.

    Args:
        dataloader (torch.utils.data.DataLoader): Dataloader of training set.
        model (hwr.model.BaseModel): Model.
        fn_loss (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        scaler (torch.cuda.amp.GradScaler): Scaler for mix-precision training.
        lr_schedular (torch.optim.lr_scheduler.SequentialLR): Learning rate scheduler.
        man (hwr.manager.RunManager): Running manager.
        epoch (int): Current epoch number.
    '''
    man.initialize_epoch(epoch, len(dataloader), False)
    model.train()

    for idx, (x, y, len_x, len_y) in enumerate(dataloader):
        x, y = x.to(man.cfgs.device), y.to(man.cfgs.device)
        optimizer.zero_grad()

        with torch.autocast('cuda', torch.float16):
            out = model(x)
            loss = fn_loss(
                out.permute((1, 0, 2)), y, len_x // model.ratio_ds, len_y
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        man.update_iteration(
            idx,
            loss.item(),
            lr_scheduler.get_last_lr()[0],
        )

    man.summarize_epoch()

    # save checkpoints every freq_save epoch
    if man.check_step(epoch + 1, 'save'):
        man.save_checkpoint(
            model.state_dict(),
            optimizer.state_dict(),
            lr_scheduler.state_dict(),
        )


def test(
    dataloader: DataLoader,
    model: BaseModel,
    fn_loss: nn.Module,
    man: RunManager,
    ctc_decoder: BestPath,
    epoch: int | None = None,
) -> None:
    '''Test the model.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader of test set.
        model (hwr.model.BaseModel): Model.
        fn_loss (torch.nn.Module): Loss function.
        man (hwr.manager.RunManager): Running manager.
        ctc_decoder (BestPath): An instance of CTC decoder.
        epoch (int | None, optional): Epoch number. Defaults to None.
    '''
    preds = []  # predictions for evaluation
    labels = []  # labels for evaluation
    man.initialize_epoch(epoch, len(dataloader), True)
    model.eval()

    with torch.no_grad():
        for idx, (x, y, len_x, len_y) in enumerate(dataloader):
            x, y = x.to(man.cfgs.device), y.to(man.cfgs.device)
            out = model(x)
            loss = fn_loss(
                out.permute((1, 0, 2)), y, len_x // model.ratio_ds, len_y
            )
            man.update_iteration(idx, loss.item())

            # decode and cache results every freq_eval epoch
            if man.check_step(epoch + 1, 'eval'):
                for pred, len_pred, label in zip(
                    out.cpu(), len_x // model.ratio_ds, y.cpu()
                ):
                    preds.append(ctc_decoder.decode(pred[:len_pred]))
                    labels.append(ctc_decoder.decode(label, True))

    man.summarize_epoch()

    # evaluate every freq_eval epoch
    if man.check_step(epoch + 1, 'eval'):
        visualize(preds, labels, man.cfgs.categories[1:], man.dir_vis, epoch)
        results_eval = evaluate(preds, labels)
        man.update_evaluation(results_eval, preds[:20], labels[:20])


def main(cfgs: argparse.Namespace) -> None:
    '''Main function for training and evaluation.

    Args:
        cfgs (argparse.Namespace): Configurations.
    '''
    # initialize the environment
    manager = RunManager(cfgs)
    seed_everything(cfgs.seed)
    ctc_decoder = BestPath(cfgs.categories)

    model = BaseModel(
        cfgs.arch_en,
        cfgs.arch_de,
        cfgs.num_channel,
        len(cfgs.categories),
        cfgs.len_seq,
    ).to(cfgs.device)
    print(model)
    dataset_test = HRDataset(
        os.path.join(cfgs.dir_dataset, 'val.json'),
        cfgs.categories,
        model.ratio_ds,
        cfgs.idx_fold,
        cfgs.len_seq,
        cache=cfgs.cache,
    )
    dataloader_test = DataLoader(
        dataset_test,
        cfgs.size_batch,
        num_workers=cfgs.num_worker,
        collate_fn=fn_collate,
    )
    fn_loss = CTCLoss()
    epoch_start = 0

    if not cfgs.test:
        dataset_train = HRDataset(
            os.path.join(cfgs.dir_dataset, 'train.json'),
            cfgs.categories,
            model.ratio_ds,
            cfgs.idx_fold,
            cfgs.len_seq,
            cfgs.aug,
            cfgs.cache,
        )
        dataloader_train = DataLoader(
            dataset_train,
            cfgs.size_batch,
            True,
            num_workers=cfgs.num_worker,
            collate_fn=fn_collate,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(cfgs.seed),
        )
        optimizer = torch.optim.AdamW(model.parameters(), cfgs.lr)
        scaler = GradScaler()
        lr_scheduler = SequentialLR(
            optimizer,
            [
                LinearLR(
                    optimizer,
                    0.01,
                    total_iters=len(dataloader_train) * cfgs.epoch_warmup,
                ),
                CosineAnnealingLR(
                    optimizer,
                    len(dataloader_train) * (cfgs.epoch - cfgs.epoch_warmup),
                ),
            ],
            [len(dataloader_train) * cfgs.epoch_warmup],
        )

    # load checkpoint if given
    if cfgs.checkpoint:
        ckp = torch.load(cfgs.checkpoint, weights_only=False)
        model.load_state_dict(ckp['model'], strict=False)

        if not cfgs.test:
            if 'epoch' in ckp.keys():  # resume
                epoch_start = ckp['epoch'] + 1
                optimizer.load_state_dict(ckp['optimizer'])
                lr_scheduler.load_state_dict(ckp['lr_scheduler'])
            elif cfgs.freeze:  # freeze
                for params in model.encoder.parameters():
                    params.requires_grad = False
            else:  # finetune
                optimizer = torch.optim.AdamW(
                    [
                        {
                            'params': model.encoder.parameters(),
                            'lr': cfgs.lr * 0.1,
                        },
                        {
                            'params': model.decoder.parameters(),
                            'lr': cfgs.lr,
                        },
                    ]
                )
                lr_scheduler = SequentialLR(
                    optimizer,
                    [
                        LinearLR(
                            optimizer,
                            0.01,
                            total_iters=len(dataloader_train)
                            * cfgs.epoch_warmup,
                        ),
                        CosineAnnealingLR(
                            optimizer,
                            len(dataloader_train)
                            * (cfgs.epoch - cfgs.epoch_warmup),
                        ),
                    ],
                    [len(dataloader_train) * cfgs.epoch_warmup],
                )

        logger.info(f'Load checkpoint from {cfgs.checkpoint}')

    # start running
    for e in range(epoch_start, cfgs.epoch):
        if cfgs.test:
            test(
                dataloader_test,
                model,
                fn_loss,
                manager,
                ctc_decoder,
                -1,
            )
            break
        else:
            train_one_epoch(
                dataloader_train,
                model,
                fn_loss,
                optimizer,
                scaler,
                lr_scheduler,
                manager,
                e,
            )
            test(
                dataloader_test,
                model,
                fn_loss,
                manager,
                ctc_decoder,
                e,
            )

    if not cfgs.test:
        manager.summarize_evaluation()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run handwriting recognition model.'
    )
    parser.add_argument(
        '-c', '--config', help='Path to YAML file of configuration.'
    )
    args = parser.parse_args()
    # args.config = 'configs/train.yaml'

    with open(args.config, 'r') as f:
        cfgs = yaml.safe_load(f)
        cfgs = argparse.Namespace(**cfgs)

    main(cfgs)
