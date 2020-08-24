'''
This file contains the utility function required for the GP equation
i.e. The frequency-domain transforms, gaussian/laplacian operators, laplaican pyramid etc.
'''

import torch
import math
import numpy as np
import scipy
import skimage


from skimage.io import imread
from skimage.transform import resize
from scipy.fftpack import dct, idct
from skimage.filters import gaussian, sobel_h, sobel_v

def transpose(np_array):
  return np.transpose(np_array, (2, 0, 1))

def untranspose(np_array):
  return np.transpose(np_array, (1, 2, 0))

def convert_range_normal(np_array):
  '''Converts to [0, 1] range from [-1, 1] range'''
  return np.clip((np_array + 1)/2, 0, 1)

def convert_range_GAN(np_array):
  '''Converts to [-1, 1] range from [0, 1] range'''
  return np.clip((np_array * 2 - 1), -1, 1)

def read_image(path):
  image = skimage.img_as_float(imread(path))
  return image.astype(np.float32)

def read_mask(path):
  return imread(path).astype(np.float32)
  
def resize_image(im, image_size, order=3, dtype=None):
  '''Resizes the image - bilinear/bicubic given by order'''
  im = resize(im, image_size, preserve_range=True, order=order, mode='constant')
  if dtype:
      im = im.astype(dtype)
  return im

def fft2(kernel, size, dtype):
  '''2D Fourier transform and returns the real part'''
  w,h = size
  transformed_kernel = np.fft.fft2(kernel)
  return np.real(transformed_kernel[0:w, 0:h]).astype(dtype) 
  
def dct2(kernel, norm = 'ortho'):
  '''2D Discrete Cosine transform'''
  return dct(dct(kernel, norm = norm).T, norm = norm).T
  
def idct2(kernel, norm = 'ortho'):
  '''2D Inverse Discrete Cosine transform'''
  return idct(idct(kernel, norm = norm).T, norm = norm).T

def laplacian_operator(size, dtype):
  '''Returns the laplacian operator in the frequency domain'''
  w,h = size
  operator = np.zeros((2 * w, 2 * h)).astype(dtype)

  laplacian_kernel = [
    [0, -1, 0 ],
    [-1, 4, -1],
    [0, -1, 0 ]
  ]

  operator[:3, :3] = laplacian_kernel
  operator = np.roll(operator, -1, axis = 0)
  operator = np.roll(operator, -1, axis = 1)

  freq_operator = fft2(operator, size, dtype)
  return freq_operator

def gaussian_operator(size, dtype, smoothing_sigma):
  '''Returns the Gaussian operator in the frequency domain'''
  w,h = size
  operator = np.zeros((2 * w, 2 * h)).astype(dtype)

  operator[1,1] = 1
  operator[:3, :3] = gaussian(operator[:3, :3], smoothing_sigma)
  operator = np.roll(operator, -1, axis = 0)
  operator = np.roll(operator, -1, axis = 1)

  freq_operator = fft2(operator, size, dtype)
  return freq_operator

def filter_2d(image, filter_func):
  '''Should be called with numpy arrays (third dimension as channel)'''
  gradients = np.zeros_like(image)
  for i in range(image.shape[2]):
    gradients[:,:,i] = filter_func(image[:,:,i])

  return gradients

def get_gradients_sobel(image):
  '''Returns the image filtered with the Sobel kernel in the horizontal axis and the vertical axis i.e. the gradients'''
  horizontal_filter, vertical_filter = sobel_h, sobel_v

  output = np.zeros((*image.shape, 2)) 

  output[:,:,:,0] = filter_2d(image, horizontal_filter)
  output[:,:,:,1] = filter_2d(image, vertical_filter)

  return output.astype(image.dtype)


def get_laplacian_pyramid(image, max_level, GAN_image_size, smoothing_sigma):
  '''Returns laplacian pyramid in the order GAN_image_size, GAN_image_size * 2, ... , image.size'''

  image_pyramid = [image]
  diff_pyramid = []

  for i in range(max_level - 1, -1, -1):
    cur_size = (GAN_image_size * (2 ** i), GAN_image_size * (2 ** i))
    smoothed = gaussian(image_pyramid[-1], smoothing_sigma, multichannel = True)
    diff = image_pyramid[-1] - smoothed
    smoothed = resize_image(smoothed, cur_size)

    image_pyramid.append(smoothed)
    diff_pyramid.append(diff)

  image_pyramid.reverse()
  diff_pyramid.reverse()

  return image_pyramid, diff_pyramid 
