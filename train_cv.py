import argparse
import json
import os

import yaml


def train_cv(cfgs: dict, path_main: str) -> None:
    '''Train model in a cross validation fashion.

    Args:
        cfgs (dict): Training configuration.
        path_main (str): Path to the Python script for training.
    '''
    with open(os.path.join(cfgs['dir_dataset'], 'train.json'), 'r') as f:
        num_fd = json.load(f)['info']['num_fold']

    dir_temp = 'temp'

    while os.path.isdir(dir_temp):
        dir_temp += '1'

    os.mkdir(dir_temp)

    command = []
    seperator = ' && '

    for i in range(num_fd):
        cfgs['idx_cv'] = i
        path_temp = os.path.join(dir_temp, f'f{i}.yaml')

        with open(path_temp, 'w') as f:
            yaml.safe_dump(cfgs, f)

        command.append(f'python {path_main} -c {path_temp}')

    command = seperator.join(command) + f' && rm -rf {dir_temp}'
    print(command)
    os.system(command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run handwriting recognition model with cross validation.'
    )
    parser.add_argument(
        '-c', '--config', help='Path to the YAML file of configuration.'
    )
    parser.add_argument(
        '-m',
        '--main',
        help='Path to the Python script for training.',
        default='main.py',
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfgs = yaml.safe_load(f)

    assert (
        cfgs['idx_cv'] == -1
    ), 'Please use cross-validation training configuration.'

    train_cv(cfgs, args.main)
