#### Data augmentation for SimCLR, same as in the original paper

import dm_pix as pix
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from torchvision import transforms
from torchvision.datasets import STL10
import matplotlib.pyplot as plt

import torch.utils.data as data
import torch 


class ContrastiveTransformations(object):

    def __init__(self, base_transforms, n_views=2):
         self.base_transforms = base_transforms
         self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]

class TestTransformations(object):
    
        def __init__(self, base_transforms, n_views=1):
            self.base_transforms = base_transforms
            self.n_views = n_views
    
        def __call__(self, x):
            return self.base_transforms(x)

class SemanticTransformations(object):
    
        def __init__(self, base_transforms, n_views=64):
            self.base_transforms = base_transforms
            self.n_patches = n_views
    
        def __call__(self, x):
            # Assuming self.n_patches is a perfect square
            patches_per_side = int(np.sqrt(self.n_patches))
            patch_height = x.shape[0] // patches_per_side
            patch_width = x.shape[1] // patches_per_side

            # Reshape and transpose to get the patches
            reshaped = x.reshape(patches_per_side, patch_height, -1, patch_width)
            transposed = reshaped.transpose(0, 2, 1, 3)
            patches = transposed.reshape(-1, patch_height, patch_width, x.shape[2])

            return [self.base_transforms(x) for x in patches]

def image_to_numpy(img):
    img = np.array(img, dtype=np.float32)
    img = img/255.
    return img

contrast_transforms = transforms.Compose([transforms.RandomResizedCrop(size=96), image_to_numpy])
test_transforms = transforms.Compose([transforms.Resize(size=124), transforms.CenterCrop(size=96), image_to_numpy])

def augment_img(rng, img):
    rngs = random.split(rng, 8)
    img = pix.random_flip_left_right(rngs[0], img)
    img_jt = img
    img_jt = img_jt * random.uniform(rngs[1], shape=(1,), minval=0.6, maxval=1.4)
    img_jt = jax.lax.clamp(0.0, img_jt, 1.0)
    img_jt = pix.random_contrast(rngs[2], img_jt, lower=0.6, upper=1.4)
    img_jt = jax.lax.clamp(0.0, img_jt, 1.0)
    img_jt = pix.random_saturation(rngs[3], img_jt, lower=0.6, upper=1.4)
    img_jt = jax.lax.clamp(0.0, img_jt, 1.0)
    img_jt = pix.random_hue(rngs[4], img_jt, max_delta=0.1)
    img_jt = jax.lax.clamp(0.0, img_jt, 1.0)
    should_jt = random.bernoulli(rngs[5], p=0.8)
    img = jnp.where(should_jt, img_jt, img)
    # Random grayscale
    should_gs = random.bernoulli(rngs[6], p=0.2)
    img = jax.lax.cond(should_gs,
                        lambda x: pix.rgb_to_grayscale(x, keep_dims=True),
                        lambda x: x,
                        img)
    # Gaussian Blur
    sigma = random.uniform(rngs[7], shape=(1,), minval=0.1, maxval=2.0)
    img = pix.gaussian_blur(img, sigma=sigma[0], kernel_size=9)
    # Normalization
    img = img * 2.0 - 1.0
    return img 

def test_augment_img(rng, img):
    return img * 2.0 - 1.0

parallel_augment = jax.jit(lambda rng, imgs: jax.vmap(augment_img)(random.split(rng, imgs.shape[0]), imgs))

test_augment = jax.jit(lambda rng, imgs: jax.vmap(test_augment_img)(random.split(rng, imgs.shape[0]), imgs))

def numpy_collate_contrastive(batch):
    imgs1, imgs2 = [[b[0][i] for b in batch] for i in range(2)]
    return np.stack(imgs1 + imgs2, axis=0)

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)



