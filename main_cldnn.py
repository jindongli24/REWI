import argparse
import os
import warnings

import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from rewi.ctc_decoder import BestPath
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
    lr_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    man: RunManager,
    epoch: int,
) -> None:
    '''Train model for 1 epoch.

    Args:
        dataloader (torch.utils.data.DataLoader): Dataloader of training set.
        model (hwr.model.BaseModel): Model.
        fn_loss (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        lr_schedular (torch.optim.lr_scheduler.ReduceLROnPlateau): Learning rate scheduler.
        man (hwr.manager.RunManager): Running manager.
        epoch (int): Current epoch number.
    '''
    man.initialize_epoch(epoch, len(dataloader), False)
    model.train()

    for idx, (x, y, len_x, len_y) in enumerate(dataloader):
        x, y = x.to(man.cfgs.device), y.to(man.cfgs.device)
        optimizer.zero_grad()
        out = model(x)
        loss = fn_loss(
            out.permute((1, 0, 2)), y, len_x // model.ratio_ds, len_y
        )
        loss.backward()
        optimizer.step()
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
        dataloader (torch.utils.data.DataLoader): DataLoader of testing set.
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

    loss_val = man.summarize_epoch()

    # evaluate every freq_eval epoch
    if man.check_step(epoch + 1, 'eval'):
        visualize(preds, labels, man.cfgs.categories[1:], man.dir_vis, epoch)
        results_eval = evaluate(preds, labels)
        man.update_evaluation(results_eval, preds[:20], labels[:20])

    return loss_val


def main(cfgs: argparse.Namespace) -> None:
    '''Main function for training and evaluation.

    Args:
        cfgs (argparse.Namespace): Configurations.
    '''
    # initialize the environment
    manager = RunManager(cfgs)
    seed_everything(cfgs.seed)
    ctc_decoder = BestPath(cfgs.categories)

    # initialize the datasets and dataloaders
    model = BaseModel(
        cfgs.arch_en,
        cfgs.arch_de,
        cfgs.num_channel,
        len(cfgs.categories),
    ).to(cfgs.device)
    dataset_test = HRDataset(
        os.path.join(cfgs.dir_dataset, 'val.json'),
        cfgs.categories,
        model.ratio_ds,
        cfgs.idx_fold,
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
            cache=cfgs.cache,
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
        optimizer = torch.optim.Adam(model.parameters(), cfgs.lr)
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.8, min_lr=1e-4)

    # start running
    losses_val = []

    for e in range(epoch_start, 1000):
        if cfgs.test:
            test(
                dataloader_test,
                model,
                fn_loss,
                manager,
                ctc_decoder,
                epoch=-1,
            )
            break
        else:
            train_one_epoch(
                dataloader_train,
                model,
                fn_loss,
                optimizer,
                lr_scheduler,
                manager,
                e,
            )
            loss_val = test(
                dataloader_test,
                model,
                fn_loss,
                manager,
                ctc_decoder,
                e,
            )
            lr_scheduler.step(loss_val)

            if lr_scheduler.get_last_lr()[0] <= 1e-4:
                if len(losses_val) >= 19:
                    if loss_val > losses_val[-19]:
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
