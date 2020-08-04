import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_network(config):
  # parse the config file, pass it to the networks and return the G and D
  image_size, num_encoder_filters, num_bottleneck, num_output_channels = config.OUTPUT_SIZE, config.NUM_FILTERS, config.NUM_BOTTLENECK, config.NUM_OUTPUT_CHANNELS
  return Generator(image_size, num_encoder_filters, num_bottleneck, num_output_channels), Discriminator(image_size, num_encoder_filters, num_bottleneck)


'''GENERIC ENCODER'''
class GenericEncoder(nn.Module):
  def __init__(self, image_size, num_encoder_filters, num_bottleneck):
    super(GenericEncoder, self).__init__()
    layers = []

    # fill

    self.net = nn.Sequential(layers)

  def forward(self, x):
    return self.net(x)

'''DECODER'''
class Decoder(nn.Module):
  def __init__(self, image_size, num_decoder_filters, num_output_channels):
    super(Decoder, self).__init__()
    layers = []

    # fill

    self.net = nn.Sequential(layers)

  def forward(self, x):
    return self.net(x)

'''DISCRIMINATOR'''
class Discriminator(nn.Module):
  def __init__(self, image_size, num_encoder_filters, num_bottleneck):
    super(Discriminator, self).__init__()
    self.net = GenericEncoder(image_size, num_encoder_filters, num_bottleneck)

  def forward(self, x):
    x = self.net(x)
    
    # see what to do here - return only one regressed value(critic)

    return x


'''GENERATOR'''
class Generator(nn.Module):
  def __init__(self, image_size, num_encoder_filters, num_bottleneck, num_output_channels):
    super(Generator, self).__init__()
    self.encoder = GenericEncoder(image_size, num_encoder_filters, num_bottleneck)
    self.bn = nn.BatchNorm2d(num_bottleneck)
    self.decoder = Decoder(image_size, num_decoder_filters, num_output_channels)

  def forward(self, x):
    x = self.encoder(x)
    x = F.leaky_relu(self.bn(x), negative_slope=0.01)
    x = self.decoder(x)
    return x 