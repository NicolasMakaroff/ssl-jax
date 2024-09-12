import jax.numpy as jnp
import jax


def target_ema(global_step: jnp.ndarray,
               base_ema: float,
               max_steps: int) -> jnp.ndarray:
  decay = _cosine_decay(global_step, max_steps, 1.)
  return 1. - (1. - base_ema) * decay


def learning_schedule(global_step: jnp.ndarray,
                      batch_size: int,
                      base_learning_rate: float,
                      total_steps: int,
                      warmup_steps: int) -> float:


  scaled_lr = base_learning_rate * batch_size / 256.
  learning_rate = (
      global_step.astype(jnp.float32) / int(warmup_steps) *
      scaled_lr if warmup_steps > 0 else scaled_lr)


  return jnp.where(
      global_step < warmup_steps, learning_rate,
      _cosine_decay(global_step - warmup_steps, total_steps - warmup_steps,
                    scaled_lr))


def _cosine_decay(global_step: jnp.ndarray,
                  max_steps: int,
                  initial_value: float) -> jnp.ndarray:

  global_step = jnp.minimum(global_step, max_steps)
  cosine_decay_value = 0.5 * (1 + jnp.cos(jnp.pi * global_step / max_steps))
  decayed_learning_rate = initial_value * cosine_decay_value
  return decayed_learning_rate

def get_shuffle_ids(bsz, key: int):
    """generate shuffle ids for ShuffleBN"""
    forward_inds = jax.random.permutation(key, bsz)
    backward_inds = jnp.zeros(bsz, dtype=jnp.int32)
    value = jnp.arange(bsz, dtype=jnp.int32)
    backward_inds.at[forward_inds].set(value)
    return forward_inds, backward_inds


