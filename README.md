# Gaussian-Poisson-GANs-For-Image-Blending

Blending composite images(copy-paste images/foreign objects in a scene) using a Wasserstein Generative Adversarial Network(GAN) and the Gaussian-Poisson equation<sup>[1]</sup>

The GAN gives a low-res blend, which is then passed to the post-hoc Gaussian-Poisson component which iteratively upsamples using the Laplacian pyramid of the object's image and the scene's image, while solving an optimization problem to estimate the low-frequency signals(i.e. using a Gaussian blur) of the GAN's output and estimating the high-frequency signals(i.e. image gradient) of the composite(copy-paste) image using the pyramid.

# Results

| Source Image    | Target Image           | Blend      |
|:---------------:|:----------------------:|:----------:|
|![](docs/source.jpg)|![](docs/dest.jpg)|![](docs/blend.png)|
|![](docs/source.jpeg)|![](docs/dest.jpeg)|![](docs/blend1.png)|


# Instructions

<details>
<summary>
Data
</summary>
<br>
  
[The Transient Attributes dataset](http://transattr.cs.brown.edu/files/aligned_images.tar) - 1.8 GB

Once it is downloaded, extract the .tar file. You will find a folder named _'imageAlignedLD'_ .

You can crop the images by executing the following command:

```
python crop_images.py --data_path path_to_imageAlignedLD_folder --output_dir path_to_output_folder
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

The pretrained model can be downloaded from [this google drive folder](https://drive.google.com/file/d/10eePae3qZEhlyoVFElpjRaHEfAOSYIXp/view?usp=sharing). The size of the model is **845 MB**. The model has been trained for **33 epochs**, and it took around **24 hours to train**.  

# Reference

The core algorithm was presented in the **ACMMM 2019 (oral) paper** titled

[1] **GP-GAN: Towards Realistic High-Resolution Image Blending**, 
    Huikai Wu, Shuai Zheng, Junge Zhang, Kaiqi Huang
    [[paper]](https://arxiv.org/pdf/1703.07195.pdf)

# Citation

Please cite the original paper if this code is useful for your research:

```
@inproceedings{wu2017gp,
  title     = {GP-GAN: Towards Realistic High-Resolution Image Blending},
  author    = {Wu, Huikai and Zheng, Shuai and Zhang, Junge and Huang, Kaiqi},
  booktitle = {ACMMM},
  year      = {2019}
}
```
