import jax
import jax.numpy as jnp
from jax.lax import stop_gradient
import haiku as hk
from src.model.resnet import ResNet18
from typing import Mapping, Union, Optional

FloatStrOrBool = Union[str, float, bool]

@jax.jit
def _get_simsiam_loss(self, output):
        """Compute loss for SimSiam."""
        z_a = l2_normalize(stop_gradient(output[0]), axis=-1)
        z_b = l2_normalize(stop_gradient(output[1]), axis=-1)
        p_a = l2_normalize(output[2], axis=-1)
        p_b = l2_normalize(output[3], axis=-1)

        loss = 0.5 * (-(p_a*z_b).sum(axis=1).mean() + -(p_b*z_a).sum(axis=1).mean()) 

        return {
            "loss": loss,
        }

class SimSIAM(hk.Module):

    def __init__(self, dim: int = 1024, embedding_size: int = 512, pred_dim: int = 512, bn_config: Mapping[str, FloatStrOrBool] = None, n_patches: int = 2):
        super().__init__()
        bn_config = dict(bn_config or {})
        bn_config.setdefault("decay_rate", 0.9)
        bn_config.setdefault("eps", 1e-5)
        bn_config.setdefault("create_scale", True)
        bn_config.setdefault("create_offset", True)
        bn_config.setdefault("scale_init", jnp.ones),
        bn_config.setdefault("offset_init", jnp.zeros),
        self.n_patches = n_patches
        self.convnet = ResNet18(num_classes=dim, bn_config=bn_config)
        #prev_dim = self.convnet.fc.weight.shape[1]
        self.projector = Projector(dim=dim, embedding_size=embedding_size, bn_config=bn_config)
        self.predictor = Predictor(dim=dim, embedding_size=pred_dim, bn_config=bn_config)
        
    def __call__(self, imgs , is_training: bool = True):
        size = imgs.shape[0]//self.n_patches  

        model_feats_a = self.convnet(imgs[:size], is_training=is_training)
        model_feats_b = self.convnet(imgs[size:], is_training=is_training)

        z_a = self.projector(model_feats_a, is_training=is_training)
        z_b = self.projector(model_feats_b, is_training=is_training)

        p_a = self.predictor(z_a, is_training=is_training)
        p_b = self.predictor(z_b, is_training=is_training)

          

        #metrics = _get_simsiam_loss(z_a, z_b, p_a, p_b)

        return z_a, z_b, p_a, p_b#metrics['loss'], metrics
    
    def encode(self, imgs, is_training: bool = False):
        model_feats = self.convnet(imgs, is_training=is_training)
        return model_feats
    
class Projector(hk.Module):

    def __init__(self, dim: int = 1024, embedding_size=512, bn_config: Mapping[str, FloatStrOrBool] = None):
        super().__init__()

        self.linear_1 = hk.Linear(dim, with_bias=True)
        self.bn_1 = hk.BatchNorm(name="bn1_projector", **bn_config)
        self.linear_2 = hk.Linear(dim, with_bias=True)
        self.bn_2 = hk.BatchNorm(name="bn2_projector", create_scale=False, create_offset=False, 
                                scale_init=None, offset_init=None, decay_rate=0.9, eps=1e-5)
        self.linear_3 = hk.Linear(dim, with_bias=False)

    def __call__(self, x, is_training: bool = True):

        x = self.linear_1(x)
        x = self.bn_1(x, is_training=is_training)
        x = jax.nn.relu(x)
        x = self.linear_2(x)
        x = self.bn_2(x, is_training=is_training)
        x = jax.nn.relu(x)
        x = self.linear_3(x)

        return x

class Predictor(hk.Module):

    def __init__(self, dim: int = 512, embedding_size: int = 512, bn_config: Mapping[str, FloatStrOrBool] = None):
        super().__init__()

        self.linear_1 = hk.Linear(dim, with_bias=True)
        self.bn_1 = hk.BatchNorm(name="bn1_predictor", **bn_config)
        self.linear_2 = hk.Linear(dim, with_bias=False)

    def __call__(self, x, is_training: bool = True):
            
            x = self.linear_1(x)
            x = self.bn_1(x, is_training=is_training)
            x = jax.nn.relu(x)
            x = self.linear_2(x)
    
            return x
    
def l2_normalize(
    x: jnp.ndarray,
    axis: Optional[int] = None,
    epsilon: float = 1e-12,
) -> jnp.ndarray:
  """l2 normalize a tensor on an axis with numerical stability."""
  square_sum = jnp.sum(jnp.square(x), axis=axis, keepdims=True)
  x_inv_norm = jax.lax.rsqrt(jnp.maximum(square_sum, epsilon))
  return x * x_inv_norm
