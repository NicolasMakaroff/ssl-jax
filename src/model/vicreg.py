import jax
import jax.numpy as jnp
import haiku as hk
from src.model.resnet import ResNet18
from typing import Mapping, Union

FloatStrOrBool = Union[str, float, bool]

@jax.jit
def _get_vicreg_loss(z_a, z_b, dim=1, eps=1e-8):
    """
    Compute the VICReg loss between two tensors.

    Args:
        x (Tensor): The first input tensor.
        y (Tensor): The second input tensor.
        dim (int, optional): The dimension along which to compute the loss. Default is 1.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-8.

    Returns:
        Tensor: The VICReg loss between x and y.

    """
    # Invariance loss
    loss_inv = jnp.mean((z_a - z_b) ** 2)

    # Variance loss
    std_z_a = jnp.sqrt(z_a.var(axis=0) + 1e-4)#hparams.variance_loss_epsilon)
    std_z_b = jnp.sqrt(z_b.var(axis=0) + 1e-4)#hparams.variance_loss_epsilon)
    loss_v_a = jnp.mean(jax.nn.relu(1 - std_z_a))
    loss_v_b = jnp.mean(jax.nn.relu(1 - std_z_b))
    loss_var = loss_v_a + loss_v_b

    # Covariance loss
    N, D = z_a.shape
    z_a_centered = z_a - jnp.mean(z_a, axis=0)
    z_b_centered = z_b - jnp.mean(z_b, axis=0)
    cov_z_a = jnp.dot(z_a_centered.T, z_a_centered) / (N - 1)
    cov_z_b = jnp.dot(z_b_centered.T, z_b_centered) / (N - 1)
    loss_c_a = (cov_z_a ** 2).sum() - jnp.diag(cov_z_a ** 2).sum()
    loss_c_b = (cov_z_b ** 2).sum() - jnp.diag(cov_z_b ** 2).sum()
    loss_cov = (loss_c_a + loss_c_b) / D

    # Weighted loss
    weighted_inv = loss_inv * 25.0 #hparams.invariance_loss_weight
    weighted_var = loss_var * 25.0 #hparams.variance_loss_weight
    weighted_cov = loss_cov * 1.0 #hparams.covariance_loss_weight

    # Total loss
    loss = weighted_var + weighted_cov + weighted_inv

    return {
        "loss": loss,
        "loss_invariance": weighted_inv,
        "loss_variance": weighted_var,
        "loss_covariance": weighted_cov,
    }

class VICReg(hk.Module):

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
        self.mu_linear = hk.Linear(512, with_bias=True)
        self.logvar_linear = hk.Linear(512, with_bias=True)
        #self.rng = jax.random.PRNGKey(42)
        #prev_dim = self.convnet.fc.weight.shape[1]
        self.projector = Projector(dim=dim, embedding_size=embedding_size, bn_config=bn_config)
        
    def __call__(self, imgs , is_training: bool = True):
        model_feats = self.convnet(imgs, is_training=is_training)
        mean = self.mu_linear(model_feats)
        var = jnp.exp(0.5*self.logvar_linear(model_feats))
        #rng, subkey = jax.random.split(self.rng)
        #jax.debug.print('subkey: {x}',x=subkey)
        epsilon = jax.random.normal(hk.next_rng_key(), shape=var.shape)
        norm = epsilon * var + mean
        z = self.projector(norm, is_training=is_training)

        size = imgs.shape[0]//self.n_patches    
        #jax.debug.print('collapse? : {x}', x=z[0])
        #jax.debug.print('collapse? : {x}', x=z[size])
        #kl_metric = jnp.mean((mean[:size] - mean[size:])**2)
        #if self.n_patches == 2:
        metrics = _get_vicreg_loss(z[:size], z[size:])
        """else:
            metrics = {'loss': 0, 'loss_invariance': 0, 'loss_variance': 0, 'loss_covariance': 0,} 
            for i in range(1, self.n_patches-1):
                tmp = _get_vicreg_loss(z[size*i:size*(i+1)], z[(i+1)*size:size*(i+2)])
                metrics['loss'] += tmp['loss'] / self.n_patches-1
                metrics['loss_invariance'] += tmp['loss_invariance'] / self.n_patches-1
                metrics['loss_variance'] += tmp['loss_variance']/ self.n_patches-1
                metrics['loss_covariance'] += tmp['loss_covariance']/ self.n_patches-1"""

        return metrics['loss'], metrics
    
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
    