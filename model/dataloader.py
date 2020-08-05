import torch
import torchvision
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import os
import glob
import math

from config import *

class BlendingDataset(torch.utils.data.Dataset):
  def __init__(self, num_samples, folders, data_dir, center_square_ratio, scaling_size, output_size):
    folder_images = {folder: glob.glob(os.path.join(data_dir, folder, '*')) for folder in folders}
    self.scaling_size = scaling_size
    self.output_size = output_size
    self.center_square_size = int(output_size * center_square_ratio)
    self.start_center_crop = self.output_size // 2 - self.center_square_size // 2

    self.samples = []
    for _ in range(num_samples):
      folder = np.random.choice(folders)
      obj_path, bg_path = np.random.choice(folder_images[folder], 2, replace=False)
      self.samples.append((obj_path, bg_path))

  def scale_and_crop(self, image, resize_w, resize_h, start_x, start_y):
    image = resize(image, (resize_w, resize_h), order=1, preserve_range=False, mode='constant')
    image = image[start_x:start_x + self.output_size, start_y:start_y + self.output_size, :] * 2 - 1 # shifts from [0,1] to [-1,1]
    return np.transpose(image, (2, 0, 1)).astype(np.float32)

  def __getitem__(self, idx):
    '''READ THE OBJECT AND THE BACKGROUND IMAGES'''
    obj_path, bg_path = self.samples[idx]
    obj = imread(obj_path); bg = imread(bg_path)

    '''WE ULTIMATELY WANT IT IN self.output_size, SO, FIND THE SCALING RATIO FROM THE MINIMUM SIZE AND THEN CROP IT TO self.output_size'''
    '''INSTEAD OF DIRECTLY RESIZING TO self.output_size, WE RESIZE TO A PARTICULAR SIZE, AND PROVIDE A START X,Y'''
    w, h, _ = obj.shape
    min_size = min(w, h)
    ratio = self.output_size / min_size
    rw, rh = int(math.ceil(w * ratio)), int(math.ceil(h * ratio))
    sx, sy = np.random.random_integers(0, rw - self.output_size), np.random.random_integers(0, rh - self.output_size)

    '''RESIZE TO rw, rh AND CROP TO OUTPUT SIZE USING sx, sy'''
    obj_cropped = self.scale_and_crop(obj, rw, rh, sx, sy)
    bg_cropped = self.scale_and_crop(bg, rw, rh, sx, sy)

    '''COPY THE CENTER OF THE OBJECT IMAGE TO THE CENTER OF THE BACKGROUND IMAGE; CENTER HERE REFERS TO STARTING POINT + CENTER CROP SIZE'''
    composite_image = bg_cropped.copy()
    composite_image[:, self.start_center_crop:self.start_center_crop + self.center_square_size, self.start_center_crop:self.start_center_crop + self.center_square_size] = obj_cropped[:,
                                                                                    self.start_center_crop:self.start_center_crop + self.center_square_size,
                                                                                    self.start_center_crop:self.start_center_crop + self.center_square_size]

    return torch.from_numpy(composite_image), torch.from_numpy(bg_cropped)


  def __len__(self):
    return len(self.samples)


class Dataloaders:
  def __init__(self, config):
    folders = sorted(
      [folder for folder in os.listdir(config.CROPPED_SAMPLES_DIR) if os.path.isdir(os.path.join(config.CROPPED_SAMPLES_DIR, folder))])
    val_end = int(len(folders) * config.VAL_RATIO)
    train_folders = folders[val_end:] 
    val_folders = folders[:val_end]
    self.config = config

    self.train_dataset = BlendingDataset(config.NUM_TRAIN_SAMPLES, train_folders, config.CROPPED_SAMPLES_DIR, config.CENTER_SQUARE_RATIO, config.SCALING_SIZE, config.OUTPUT_SIZE)
    self.val_dataset = BlendingDataset(config.NUM_VAL_SAMPLES, val_folders, config.CROPPED_SAMPLES_DIR, config.CENTER_SQUARE_RATIO, config.SCALING_SIZE, config.OUTPUT_SIZE)

  def get_train_dataloader(self):
    return torch.utils.data.DataLoader(self.train_dataset, batch_size = self.config.TRAIN_BATCH_SIZE, shuffle = self.config.TRAIN_SHUFFLE, num_workers = self.config.TRAIN_NUM_WORKERS)

  def get_val_dataloader(self):
    return torch.utils.data.DataLoader(self.val_dataset, batch_size = self.config.VAL_BATCH_SIZE, shuffle = self.config.VAL_SHUFFLE, num_workers = self.config.VAL_NUM_WORKERS)