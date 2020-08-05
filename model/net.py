import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


'''UTILITY FUNCTIONS'''
def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_network(config):
  '''Parse the config file, pass it to the networks and return the G and D'''
  image_size, num_encoder_filters, num_bottleneck, num_output_channels = config.OUTPUT_SIZE, config.NUM_FILTERS, config.NUM_BOTTLENECK, config.NUM_OUTPUT_CHANNELS
  generator = Generator(image_size, num_encoder_filters, num_bottleneck, num_output_channels) 
  discriminator = Discriminator(image_size, num_encoder_filters, num_bottleneck)
  
  generator.apply(weights_init)
  discriminator.apply(weights_init)

  return generator, discriminator


'''GENERIC ENCODER'''
class GenericEncoder(nn.Module):
  def __init__(self, image_size, num_encoder_filters, num_bottleneck):
    super(GenericEncoder, self).__init__()
    layers = []

    
    cur_image_size = image_size
    in_channels = 3
    out_channels = num_encoder_filters


    while(cur_image_size > 4):
      layers.append(nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias = False))
      layers.append(nn.BatchNorm2d(out_channels))
      layers.append(nn.LeakyReLU())

      cur_image_size /= 2
      in_channels = out_channels
      out_channels *= 2

    layers.append(nn.Conv2d(out_channels//2, num_bottleneck, 4, stride=1, padding=0, bias = False))


    self.net = nn.Sequential(*layers)

  def forward(self, x):
    # CHECK AGAIN, AS THIS IS TREATED AS AN FC LAYER
    return self.net(x)

'''DECODER'''
class Decoder(nn.Module):
  def __init__(self, image_size, num_decoder_filters, num_bottleneck, num_output_channels):
    super(Decoder, self).__init__()
    layers = []

    cur_num_channels = num_decoder_filters // 2 # done so that it is consistent with the cur_image_size 
    cur_image_size = image_size

    while cur_image_size != 4:
      cur_num_channels *= 2
      cur_image_size /= 2

    in_channels = num_bottleneck
    out_channels = cur_num_channels


    layers.append(nn.ConvTranspose2d(in_channels, out_channels, 4, stride=1, padding=0, bias = False))
    layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU())

    in_channels = out_channels
    out_channels = out_channels // 2

    while(cur_image_size < image_size // 2):
      layers.append(nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias = False))
      layers.append(nn.BatchNorm2d(out_channels))
      layers.append(nn.ReLU())

      cur_image_size *= 2
      in_channels = out_channels
      out_channels = out_channels // 2

    layers.append(nn.ConvTranspose2d(out_channels*2, num_output_channels, 4, stride=2, padding=1, bias = False))
    layers.append(nn.Tanh())     

    self.net = nn.Sequential(*layers)

  def forward(self, x):
    return self.net(x)

'''DISCRIMINATOR'''
class Discriminator(nn.Module):
  def __init__(self, image_size, num_encoder_filters, num_bottleneck):
    super(Discriminator, self).__init__()
    self.net = GenericEncoder(image_size, num_encoder_filters, num_bottleneck)

  def forward(self, x):
    x = self.net(x)   

    return x.squeeze() # returns only batch_size values(the critic value by the WGAN)


'''GENERATOR'''
class Generator(nn.Module):
  def __init__(self, image_size, num_encoder_filters, num_decoder_filters, num_bottleneck, num_output_channels):
    super(Generator, self).__init__()
    self.encoder = GenericEncoder(image_size, num_encoder_filters, num_bottleneck)
    self.bn = nn.BatchNorm2d(num_bottleneck)
    self.decoder = Decoder(image_size, num_decoder_filters, num_bottleneck, num_output_channels)

  def forward(self, x):
    x = self.encoder(x)
    x = F.leaky_relu(self.bn(x), negative_slope=0.01)
    x = self.decoder(x)
    return x 