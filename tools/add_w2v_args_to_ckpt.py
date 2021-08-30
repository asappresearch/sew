# Copyright (c) ASAPP Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# this script fixes the bug that fine-tuned model doesn't 
# save the config of the pre-trained encoder in the checkpoint

import fire
import os
from tqdm.auto import tqdm
import torch

def add_w2v_args(filename, skip=False, new_suffix='.pt'):
    print(f'processing {filename}')
    ft_ckpt = torch.load(filename, map_location='cpu')
    if not 'w2v_path' in ft_ckpt['cfg']['model']:
        print('not a fine-tuned model')
        return
    elif skip and ft_ckpt['cfg']['model'].get('w2v_args', None) is not None:
        print('already done, skipped')
        return
    pt_ckpt = torch.load(ft_ckpt['cfg']['model']['w2v_path'], map_location='cpu')
    ft_ckpt['cfg']['model']['w2v_args'] = pt_ckpt['cfg']
    new_filename = filename.replace('.pt', new_suffix)
    torch.save(ft_ckpt, new_filename)
    print(f'saved to {new_filename}')


def main(root="save-ft", suffix='.pt', dry_run=False, skip=True, new_suffix='.pt'):
    filenames = []
    for dirname, dirs, files in tqdm(os.walk(root)):
        for filename in files:
            if filename.endswith(suffix):
                filenames.append(os.path.join(dirname, filename))

    print('checkpoints:')
    print('\n'.join(filenames))

    if dry_run:
        return
    
    for filename in filenames:
        add_w2v_args(filename, skip=skip)


if __name__ == "__main__":
    fire.Fire(main)

