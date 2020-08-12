# Gaussian-Poisson-GANs-For-Image-Blending

Blending composite images(copy-paste images/foreign objects in a scene) using a Wasserstein Generative Adversarial Network(GAN) and a Gaussian-Poisson equation with a Laplacian Pyramid.

The GAN gives a low-res blend, which is then passed to the post-hoc Gaussian-Poisson component which iteratively upsamples using the Laplacian pyramid of the object's image and the scene's image, while solving an optimization problem to estimate the low-frequency signals(i.e. using a Gaussian blur) of the GAN's output and estimating the high-frequency signals(i.e. image gradient) of the composite(copy-paste) image using the pyramid.


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
