from importlib.metadata import distributions

import numpy as np
import torch
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def gen_general(n, phase='test', n_data=32768 * 2, grid_step=0.01):
    dir = f'./data/bidder={n}/{phase}'
    if not os.path.exists(dir):
        os.makedirs(dir)
    value = torch.rand(n_data, n)
    bid = torch.rand(n_data, n)
    if phase == 'test':
        bid = torch.round(bid / grid_step) * grid_step
    budget = 0.25 * n + (0.5 * n - 0.25 * n) * torch.rand(n_data)

    np.save(os.path.join(dir, 'value'), value.numpy())
    np.save(os.path.join(dir, 'bid'), bid.numpy())
    np.save(os.path.join(dir, 'budget'), budget.numpy())


if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    distributions='general'#'asymmetry','irregular'
    a=[15,20,25,30,35]
    if distributions=='general':
        for n_bidder in a:
            gen_general(n_bidder, phase='test', n_data=32768 )
            gen_general(n_bidder, phase='final_test', n_data=32768 )
            print("generated general n_bidder", n_bidder)

