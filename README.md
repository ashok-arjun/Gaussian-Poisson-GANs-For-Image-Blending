# Gaussian-Poisson-GANs-For-Image-Blending

Blending composite images(copy-paste images/foreign objects in a scene) using a Wasserstein Generative Adversarial Network(GAN) and a Gaussian-Poisson equation with a Laplacian Pyramid.

The GAN gives a low-res blend, which is then passed to the post-hoc Gaussian-Poisson component which iteratively upsamples using the Laplacian pyramid of the object's image and the scene's image, while solving an optimization problem to estimate the low-frequency signals(i.e. using a Gaussian blur) of the GAN's output and estimating the high-frequency signals(i.e. image gradient) of the composite(copy-paste) image using the pyramid.

# Results

(1-2-3 image)

# Instructions

<details>
<summary>
Data
</summary>
<br>
  
[The Transient Attributes dataset](http://transattr.cs.brown.edu/files/aligned_images.tar) - 1.8 GB

Once it is downloaded, extract the .tar file and crop the images by executing the following code:

```
from crop_images import crop_images

crop_images('path_to_imageAlignedLD_folder', 'path_to_result_folder')
```
<br>
</details>
<details>

<summary>
Training
</summary>

</details>

<details>

<summary>
Inference
</summary>
</details>

# Pretrained model

The pretrained model can be downloaded from [this https url](https://storage.googleapis.com/kagglesdsdata/datasets%2F849811%2F1449746%2F33_epochs.pth.tar?GoogleAccessId=gcp-kaggle-com@kaggle-161607.iam.gserviceaccount.com&Expires=1598963565&Signature=ks%2FHLeveIy6l27OV8mlrNhnV3fL%2B9rewKGrsKREY88HRzZWdzdrEAYC%2FAPd0uBpRpTiXaXfTWqfPgc9SR1rg59jRjcW8Rg9W1PXtY7Ae%2F1G%2BjfFN%2FTsVBotoIjg4F8Loejl8DskFI9m2taEns5pAY9N5PNqjtazRR63Pht3vGiiSMf%2FY%2BhZD2DgCTxf7TsEzuwFtXY91sOCU4tHzjI04wisR%2F9mEHv8jiZsuptOEkvsWh2b02kT5p5FYDGy0fGeSQ7VukXZVd1MsCSS%2F5mV61SLmUoecPUtm%2FhLR0PyYpPAGAXsUsnbfrYk%2FU%2FUuDDUlhyynsdFFwSWARZxiBUwCWg%3D%3D)

# Reference

The core algorithm was presented in the ACMMM 2019 paper titled

**GP-GAN: Towards Realistic High-Resolution Image Blending**, 
Huikai Wu, Shuai Zheng, Junge Zhang, Kaiqi Huang

[[paper]](https://arxiv.org/pdf/1703.07195.pdf)
