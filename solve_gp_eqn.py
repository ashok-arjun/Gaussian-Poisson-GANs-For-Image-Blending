'''
Solves the GP equation using the trained GAN, source image(object), destination image(bg) and the mask
Uses the laplacian pyramid, and solves the GP equation(color + gradients) at every stage of the pyramid
'''

import torch
import math
import numpy as np
import matplotlib.pyplot as plt

from vision_utils import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def solve_equation(src, dest, mask, blended_image, color_weight, gaussian_sigma):
  '''
  
  1. THE GRADIENTS OF SRC AND DEST ARE FOUND
  2. THE COMPOSITE GRADIENTS ARE FOUND
  3. THE LAPLACIAN AND GAUSSIAN OPERATORS ARE OBTAINED FOR THE GIVEN SIZE(IN THE FREQUENCY DOMAIN)
  4. THE G-P EQUATION IS SOLVED FOR **EACH CHANNEL** BY USING THE DIVERGENCE OF THE COMPOSITE GRADIENTS(derivative of the gradients) 
     AND THE PREVIOUS BLENDED IMAGE'S INTENSITY VALUES(IN THE FREQUENCY DOMAIN) 
  
  '''

  '1'
  src_gradients = get_gradients_sobel(src)
  dest_gradients = get_gradients_sobel(dest)
  
  '2'
  composite_gradients = src_gradients * mask[:, :, np.newaxis, np.newaxis] + dest_gradients * (1 - mask[:, :, np.newaxis, np.newaxis])

  '3'
  size, dtype = composite_gradients.shape[:2], composite_gradients.dtype
  lap = laplacian_operator(size, dtype)
  gauss = gaussian_operator(size, dtype, gaussian_sigma) 

  '4'

  composite_gradients_H = composite_gradients[:,:,:,0]
  composite_gradients_V = composite_gradients[:,:,:,1]
  divergence_v = \
  (np.roll(composite_gradients_H, 1, axis = 1)-composite_gradients_H) + \
  (np.roll(composite_gradients_V, 1, axis = 0)-composite_gradients_V)

  lap_gauss_operator = lap + color_weight*gauss

  result = np.zeros(blended_image.shape)

  for chnl in range(result.shape[2]):
    result[:,:,chnl] = idct2((dct2(divergence_v[:,:,chnl]) + color_weight * dct2(blended_image[:,:,chnl]))/lap_gauss_operator)

  result = np.clip(result, 0, 1)

  return result

def get_blended_image(src_path, dest_path, mask_path, G, GAN_input_size, color_weight = 1, gaussian_filter_sigma = 0.5, laplacian_gaussian_sigma = 1, direct_flag = False):
  
  '''READ IMAGES IN [0,1] RANGE'''
  src = read_image(src_path)
  dest = read_image(dest_path)
  mask = read_mask(mask_path)  
  
  w_image, h_image, _ = src.shape

  '''CONSTRUCT LAPLACIAN(GAUSSIAN) PYRAMIDS'''

  max_pyramid_level = int(math.ceil(np.log2(max(w_image, h_image)/GAN_input_size))) # this is max power of 2 in the laplacian pyramid, before the original image size
  src_pyramid, _ = get_laplacian_pyramid(src, max_pyramid_level, GAN_input_size, laplacian_gaussian_sigma)
  dest_pyramid, _ = get_laplacian_pyramid(dest, max_pyramid_level, GAN_input_size, laplacian_gaussian_sigma)

  '''GET COPY PASTE IMAGE AS INPUT FOR THE GAN'''
  mask_2d = resize_image(mask, (GAN_input_size, GAN_input_size), order = 0)[:, :, np.newaxis] # order is 0 as the image is a bool
  composite_image = src_pyramid[0] * mask_2d + dest_pyramid[0] * (1 - mask_2d)
  composite_image = torch.from_numpy(transpose(convert_range_GAN(composite_image))).unsqueeze(0).to(device)


  '''RUN THE GAN'''
  G.to(device); G.eval()
  with torch.no_grad():
    blended_image = G(composite_image)
  blended_image = convert_range_normal(untranspose(blended_image.cpu().squeeze().numpy())) # back to [0,1] range

  if direct_flag: blended_image = convert_range_normal(untranspose(composite_image.cpu().squeeze().numpy()))
  '''SOLVE THE G-P EQUATION AT EACH STAGE OF THE LAPLACIAN PYRAMID'''
  # AT EVERY LEVEL, THE GAN IMAGE(BLENDED IMAGE) AND THE MASK IMAGE SHOULD BE RESIZED TO THE LEVEL'S SIZE(THE SRC AND DEST ARE ALREADY IN THAT SIZE,
  # GIVEN BY THE PYRAMID )

  for level in range(max_pyramid_level + 1): # plus 1 for the actual image, alternatively this can be range(len(src_pyramid))
    cur_size = src_pyramid[level].shape[:2]
    cur_mask = resize_image(mask, cur_size, order = 0)
    blended_image = resize_image(blended_image, cur_size, order = 3)
    blended_image = solve_equation(src_pyramid[level], dest_pyramid[level], cur_mask, blended_image, color_weight, gaussian_filter_sigma)

  blended_image = np.clip(blended_image * 255, 0, 255).astype(np.uint8)

  return blended_image
