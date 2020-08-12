# TODO: wandb logging of losses, averaging the losses

import time
import datetime
import pytz 

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb

from model.dataloader import Dataloaders
from model.net import get_network
from model.loss import G_loss, D_loss

from test import test_GAN
from utils import *

class Trainer:
  def __init__(self, config):
    self.dataloaders = Dataloaders(config)
    self.config = config

  def train(self):
    config = self.config
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    '''DATALOADERS'''
    train_dataloader = self.dataloaders.get_train_dataloader()
    num_train_batches = len(train_dataloader)
    val_dataloader = self.dataloaders.get_val_dataloader()
    num_val_batches = len(val_dataloader)

    '''NETWORK'''
    G, D = get_network(config)
    G = G.to(device); D = D.to(device)
    
    optim_G = optim.Adam(params = G.parameters(), lr = config.G_LR, betas=(config.ADAM_BETA1, 0.999)) 
    optim_D = optim.Adam(params = D.parameters(), lr = config.G_LR, betas=(config.ADAM_BETA1, 0.999))

    G.train(); D.train()
    print('Training..')

    '''TRAINING'''
    g_iters = 0
    for epoch in range(config.START_EPOCH,config.NUM_EPOCHS):
      G.train() # since this is switched to eval in the test_GAN() method
      data_iter = iter(train_dataloader)
      batch_index = 0

      while(batch_index < num_train_batches):
        ###########################
          # (1) Update D 
        ###########################
        num_d_iters = 20 if g_iters < 25 or g_iters % 500 == 0 else config.D_ITERS   # CONFIRM - SHOULD BE 100
        d_iter = 0

        while(d_iter < num_d_iters and batch_index < num_train_batches):
          optim_D.zero_grad()
                  
          composite, bg = next(data_iter)
          composite = torch.autograd.Variable(composite.to(device)); bg = torch.autograd.Variable(bg.to(device));

          G_z = G(composite); D_G_z = D(G_z); D_x = D(bg)
          
          discrim_loss, D_real_loss, D_fake_loss = D_loss(D_x, D_G_z) # this is the critic difference(encapsulated as a combined loss)
          discrim_loss.backward()
          optim_D.step()

          for disc_param in D.parameters():
            disc_param.data.clamp_(config.D_CLAMP_RANGE[0], config.D_CLAMP_RANGE[1]) # ENFORCE LIPSCHITZ CONSTRAINT BY CLIPPING WEIGHTS

          d_iter += 1; batch_index += 1

        ###########################
          # (1) Update G 
        ###########################

        if batch_index == num_train_batches:
          break

        optim_G.zero_grad()

        composite, bg = next(data_iter)
        composite = torch.autograd.Variable(composite.to(device)); bg = torch.autograd.Variable(bg.to(device));

        G_z = G(composite)

        D_G_z = D(G_z)
        
        gen_loss, l2_loss, _ = G_loss(bg, G_z, D_G_z) # this is L2 loss between the blend and the background, with the adversarial loss
        gen_loss.backward()
        optim_G.step()

        g_iters += 1; batch_index += 1

        if g_iters % config.PRINT_EVERY == 0: 
          print(datetime.datetime.now(pytz.timezone('Asia/Kolkata')), end = ' ')
          print('Epoch: %d[%d/%d]; G_iters: %d; G_l2_loss: %f; D_real_loss: %f; D_fake_loss: %f'
          % (epoch, batch_index, num_train_batches, g_iters, l2_loss.item(), D_real_loss.item(), D_fake_loss.item())) 
          wandb.log({'Generator L2 loss': l2_loss.item()}, step = g_iters)    
          wandb.log({'Discriminator Real loss': D_real_loss.item()}, step = g_iters)    
          wandb.log({'Fake Loss': D_fake_loss.item()}, step = g_iters)    

      # END OF EPOCH
      print('-----End of Epoch: %d; G_iters: %d; G_l2_loss: %f; D_real_loss: %f; D_fake_loss: %f-----'
          % (epoch, g_iters, l2_loss.item(), D_real_loss.item(), D_fake_loss.item()))
      print('Validating...')
      destinations, composites, predicted_blends = test_GAN(G, self.dataloaders, config)
      grids = get_k_random_grids(destinations, composites, predicted_blends, k = config.LOGGING_K)
      log_images(grids, g_iters)

def log_images(images, wandb_step):
  '''
  Prints/Logs the PIL Images one by one
  '''
  for image in images:
    plt.figure()
    plt.imshow(image)     

  wandb.log({'Validation images': [wandb.Image(image) for image in images]}, step = wandb_step)      