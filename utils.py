import numpy as np
import os
import torch
import wandb

class RunningAverage():
  def __init__(self):
    self.count = 0
    self.sum = 0

  def update(self, value, n_items = 1):
    self.sum += value * n_items
    self.count += n_items

  def __call__(self):
    return self.sum/self.count  


def save_checkpoint(state, checkpoint_dir, save_to_cloud = False):
    filename = 'iter_%d.pth.tar' % (state['iteration'])
    torch.save(state, os.path.join(checkpoint_dir, filename))    
    if save_to_cloud:
      torch.save(state, os.path.join(wandb.run.dir, filename))
      wandb.save(filename)

def load_checkpoint(checkpoint, G, D, optimizer=None):
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    G.load_state_dict(checkpoint['G'])
    D.load_state_dict(checkpoint['D'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint