import glob
import os

from skimage.io import imread, imsave

def crop_images(data_dir, result_dir):

  if not os.path.isdir(result_dir):
    os.makedirs(result_dir)
  print('Cropped images will be saved to {} ...\n'.format(result_dir))

  with open('data/bbox.txt') as f:
    for line in f:
      name, bbox = line.strip().split(':')
      sx, sy, ex, ey = [int(i) for i in bbox.split(',')]

      print('Processing {} ...'.format(name))
      images = glob.glob(os.path.join(data_dir, name, '*'))
      if not os.path.isdir(os.path.join(result_dir, name)):
        os.makedirs(os.path.join(result_dir, name))

      for image in images:
        full_image = imread(image)
        cropped_image = full_image[sx:ex, sy:ey]

        imsave(os.path.join(result_dir, name, os.path.basename(image)), cropped_image)