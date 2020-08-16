import numpy as np
import os
import torch
import wandb
from PIL import Image

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
  filename = 'last.pth.tar'
  # torch.save(state, os.path.join(checkpoint_dir, filename))    
  if save_to_cloud:
    torch.save(state, os.path.join(wandb.run.dir, filename))
    wandb.save(filename)

def load_checkpoint(checkpoint, G, D):
  if not os.path.exists(checkpoint):
      raise("File doesn't exist {}".format(checkpoint))
  checkpoint = torch.load(checkpoint)
  print('Loading checkpoint from %d generator iterations'%(checkpoint['iteration']))
  G.load_state_dict(checkpoint['G'])
  D.load_state_dict(checkpoint['D'])

  return checkpoint

def get_k_random_grids(destinations, composites, predicted_blends, k = 5):
  '''
  Gets k random indices, converts the Tensor images to PIL Images/numpy arrays,
  stacks the dest - composite - blend horizontally, and returns a list of k images to be displayed
  '''

  num_images = composites.shape[0]
  indices = np.random.choice(num_images, k)

  grids = []

  with torch.no_grad():
    for idx in indices:
      composite = (np.transpose(composites[idx].cpu().numpy(), (1,2,0)) + 1) * 0.5
      destination = (np.transpose(destinations[idx].cpu().numpy(), (1,2,0)) + 1) * 0.5
      predicted_blend = (np.transpose(predicted_blends[idx].cpu().numpy(), (1,2,0)) + 1) * 0.5

      grid = np.hstack([destination, composite, predicted_blend])
      grids.append(grid)

  return grids