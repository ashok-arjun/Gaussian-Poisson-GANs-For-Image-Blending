import torch

def test_GAN(G, dataloaders, config):
  '''
  Generates the blended images for the images in the validation set, using the GAN.
  Returns the blended images, the original composite images, and the destination images.
  '''

  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

  val_dataloader = dataloaders.get_val_dataloader()
  num_val_batches = len(val_dataloader)

  G = G.to(device)
  G.eval()

  predictions = []; composites = []; destinations = [];

  for iteration, batch in enumerate(val_dataloader):

    composite_images, destination_images = batch
    composite_images = torch.autograd.Variable(composite_images.to(device))
    destination_images = torch.autograd.Variable(destination_images.to(device))
    composites.append(composite_images); destinations.append(destination_images)


    pred_blended_images = G(composite_images)
    predictions.append(pred_blended_images)

  return torch.cat(destinations, dim = 0), torch.cat(composites, dim = 0), torch.cat(predictions, dim = 0) 