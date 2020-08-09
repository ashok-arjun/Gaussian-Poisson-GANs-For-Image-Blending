import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

mse_loss = nn.MSELoss(reduction = 'mean') # Initialized only once, used in G_loss infinitely


def G_loss(x_dest, G_out, D_out, l2_weight = 0.99):
  '''
  G_loss = L2_loss(G(z), x_dest) + D(G(z))
  '''
  l2_loss = l2_weight * mse_loss(G_out, x_dest)
  adv_loss = (1-l2_weight) * D_out.mean()

  return l2_loss + adv_loss, l2_loss, adv_loss

def D_loss(D_x, D_G_z):
  '''
  D_loss = L_adv(x, G(z))
  where L_adv = x - G(z) (critic, as in WGANs, which minimizes loss for original samples and maximizes for generated samples)
  '''
  D_x_mean = D_x.mean()
  D_G_z_mean = D_G_z.mean()
  return D_x_mean - D_G_z_mean, D_x_mean, D_G_z_mean  # mean taken as (batch_size, ) Tensor is returned
