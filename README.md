# Gaussian-Poisson-GANs-For-Image-Blending

*Abstract*

# Instructions

<details>
<summary>
Data
</summary>

[The Transient Attributes dataset](http://transattr.cs.brown.edu/files/aligned_images.tar)

Once it is downloaded, extract the .tar file and crop the images by executing the following code:

```
from crop_images import crop_images

crop_images('path_to_imageAlignedLD_folder', 'path_to_result_folder')
