import glob
import os

from skimage.io import imread, imsave


not_equal = [
'90000012',
'00018680',
'00011616',
'00010098',
'00018726',
'00019435',
'00019919',
'90000002',
'90000014',
'90000004',
'00018997',
'00010780',
'00017709',
'00018964',
'00022785',
'00018515',
'00018897',
'00000384',
'00017613',
'00000398',
'00017609',
'00019906',
'90000013',
'90000007',
'00018086',
'00019000',
'90000005',
'90000006'
]

def crop_images(data_dir, result_dir):

  if not os.path.isdir(result_dir):
    os.makedirs(result_dir)
  print('Cropped images will be saved to {} ...\n'.format(result_dir))

  with open('data/bbox.txt') as f:
    for line in f:
      name, bbox = line.strip().split(':')
      sx, sy, ex, ey = [int(i) for i in bbox.split(',')]

      if name not in not_equal: continue

      print('Processing {} ...'.format(name))
      images = glob.glob(os.path.join(data_dir, name, '*'))
      if not os.path.isdir(os.path.join(result_dir, name)):
        os.makedirs(os.path.join(result_dir, name))

      for image in images:
        full_image = imread(image)
        cropped_image = full_image[sx:ex, sy:ey]

        imsave(os.path.join(result_dir, name, os.path.basename(image)), cropped_image)