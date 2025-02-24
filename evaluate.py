import argparse
import json
import os
from glob import glob

import numpy as np
import torch
import yaml
from thop import profile

from hwr.model import BaseModel


def get_mean_std_cv(cfgs: dict, results: dict = {}) -> dict:
    '''Calculate the mean and standard deviation of the results of cross
    validation.

    Args:
        cfgs (dict): Configurations.
        results (dict, optional): Current results. Defaults to {}.

    Returns:
        dict: Updated results.
    '''
    cer, wer = {}, {}

    if paths_result := glob(
        os.path.join(
            cfgs['dir_work'],
            '*',
            'test_*.json' if cfgs['test'] else 'train_*.json',
        )
    ):
        for i, path_result in enumerate(sorted(paths_result)):
            with open(path_result, 'r') as f:
                result_fd = json.load(f)
            
            if cfgs['test']:
                result_best = result_fd['-1']['evaluation']
            else:
                epoch_best = result_fd['best']['character_error_rate'][0]
                result_best = result_fd[str(epoch_best)]['evaluation']                

            cer[str(i)] = result_best['character_error_rate']
            wer[str(i)] = result_best['word_error_rate']

        results['cer'] = {
            'raw': cer,
            'mean': np.mean(list(cer.values())).item(),
            'std': np.std(list(cer.values())).item(),
        }
        results['wer'] = {
            'raw': wer,
            'mean': np.mean(list(wer.values())).item(),
            'std': np.std(list(wer.values())).item(),
        }
        results = {k: v for k, v in sorted(results.items())}

    return results


def get_macs_params(cfgs: dict, results: dict = {}) -> dict:
    '''Calcualte the number of parameters and multiply-accumulate operations
    of the network.

    Args:
        cfgs (dict): Configurations.
        results (dict, optional): Current results. Defaults to {}.

    Returns:
        dict: Updated results.
    '''
    model = BaseModel(
        cfgs['arch_en'],
        cfgs['arch_de'],
        cfgs['in_chan'],
        cfgs['num_cls'],
        cfgs['ratio_ds'],
        cfgs['len_seq'],
    ).eval()
    x = torch.randn(1, cfgs['in_chan'], 1024)
    macs, params = profile(model, inputs=(x,))

    results['macs'] = int(macs)
    results['params'] = int(params)
    results = {k: v for k, v in sorted(results.items())}

    return results


def main(path_cfg: str) -> None:
    '''Evaluate and summarize the results of all cross-validation foldes.

    Args:
        path_cfg (str): Path to the configuration YAML file.
    '''    
    with open(path_cfg, 'r') as f:
        cfgs = yaml.safe_load(f)

        if not 'ratio_ds' in cfgs.keys():
            cfgs['ratio_ds'] = 8

        if not 'len_seq' in cfgs.keys():
            cfgs['len_seq'] = 0

    if not os.path.isdir(cfgs['dir_work']):
        os.mkdir(cfgs['dir_work'])

    if os.path.isfile(os.path.join(cfgs['dir_work'], 'results.json')):
        with open(os.path.join(cfgs['dir_work'], 'results.json'), 'r') as f:
            results = json.load(f)
    else:
        results = {}

    results = get_mean_std_cv(cfgs, results)
    results = get_macs_params(cfgs, results)

    with open(os.path.join(cfgs['dir_work'], 'results.json'), 'w') as f:
        json.dump(results, f)

    print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate handwriting recognition model.'
    )
    parser.add_argument(
        '-c', '--config', help='Path to YAML file of configuration.'
    )
    args = parser.parse_args()

    main(args.config)
