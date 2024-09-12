import jax
import jax.numpy as jnp
from jax.lax import stop_gradient
import haiku as hk
from src.model.resnet import ResNet18
import optax
from src.train.trainer_module import TrainerModule, TrainState
from src.dataloader.augment import parallel_augment, image_to_numpy, test_augment
import numpy as np
from typing import Mapping, Union
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from src.model.vicreg import VICReg
from src.model.simsiam import _get_simsiam_loss
from src.model.byol import _get_byol_loss
from src.utils.schedules import target_ema

FloatStrOrBool = Union[str, float, bool]


class Trainer(TrainerModule):

    def __init__(self, model_name: str = 'vicreg', model_class = VICReg, **kwargs):
        super(Trainer, self).__init__(model_name = model_name,
                         model_class = model_class,
                         **kwargs)
        self.sklearn_classifier = LogisticRegression(max_iter=100, solver="liblinear")
        #self.hparams = attr.asdict(hparams)

    def create_functions(self):
        def loss_fn(params, target_params, state, target_state, loss_scale, rng, batch, is_training):
            batch = parallel_augment(rng, batch)
            size = batch.shape[0]//2
            online_network_out, online_state = self.forward.apply(
                params=params,
                state=state,
                rng=rng,
                batch=batch[:size],
                is_training=True)
            target_network_out, target_state = self.forward.apply(
                params=target_params,
                state=target_state,
                rng=rng,
                batch=batch[size:],
                is_training=True)
            loss, metrics = _get_byol_loss(online_network_out, stop_gradient(target_network_out))

            return loss, (metrics, dict(online_state=online_state, target_state=target_state))
        

        def train_step(train_state, batch, epoch_idx):
            params, target_params, state, target_state, opt_state, loss_scale, rng, _, _ = train_state
            #loss_fn = lambda params: loss_function(params, state.batch_stats, state.rng, batch, is_training=True)
            
            grads, (metrics, new_state) = (
                jax.grad(loss_fn, has_aux=True)(params, target_params, state, target_state, loss_scale, rng, batch, is_training=True))

            # Grads are in "param_dtype" (likely F32) here. We cast them back to the
            # compute dtype such that we do the all-reduce below in the compute precision
            # (which is typically lower than the param precision).
            policy = self.precision_policy()
            grads = policy.cast_to_compute(grads)
            grads = loss_scale.unscale(grads)

            # Taking the mean across all replicas to keep params in sync.

            #grads = jax.lax.pmean(grads, axis_name='i')

            # We compute our optimizer update in the same precision as params, even when
            # doing mixed precision training.
            grads = policy.cast_to_param(grads)

            # Compute and apply updates via our optimizer.
            updates, new_opt_state = self.optimizer().update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            rng, next_rng_key = jax.random.split(rng)
            tau = target_ema(epoch_idx,base_ema=self._base_target_ema, max_steps=self._max_steps)
            target_params = jax.tree_map(lambda x, y: x + (1 - tau) * (y - x), target_params, params)
                
            self.state = TrainState(new_params, target_params, new_state['online_state'], new_state['target_state'], new_opt_state, loss_scale, rng=next_rng_key)
            
            
            return self.state, metrics, grads

        def eval_step(train_state, batch):
            _, (metrics, _) = loss_fn(train_state.params, train_state.target_params, train_state.state, train_state.target_state, train_state.loss_scale, train_state.rng, batch, is_training=True)
            return metrics

        return train_step, eval_step
    
    def on_validation_epoch_end(self, epoch_idx, eval_metrics, val_loader):
        embedding_list = []
        labels_list = []
        for batch, labels in val_loader:
            #batch, labels = next(iter(val_loader))
            batch = test_augment(jax.random.PRNGKey(0), batch)
            embedding_list.append(np.array(self.forward.apply(self.state.params, self.state.state, None, batch, is_training=False)[0]))
            labels_list.append(labels)
        embeddings = np.concatenate([x for x in embedding_list])    
        labels = np.concatenate([x for x in labels_list])
        #jnp.save(f"embeddings_{epoch_idx}.npy", embeddings)
        num_split_linear = embeddings.shape[0] // 2
        self.sklearn_classifier.fit(embeddings[:num_split_linear], labels[:num_split_linear])
        train_accuracy = self.sklearn_classifier.score(embeddings[:num_split_linear], labels[:num_split_linear]) * 100
        valid_accuracy = self.sklearn_classifier.score(embeddings[num_split_linear:], labels[num_split_linear:]) * 100
        if epoch_idx % 10 == 0:
            #self.logger.log_model(self.forward, self.state.params, self.state.state, batch)
            X_embedded = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=30).fit_transform(embeddings)
            fig = plt.figure(figsize=(10,10))
            plt.scatter(X_embedded[:,0], X_embedded[:,1], c=labels, cmap='tab10')
            plt.savefig("asset/"+str(self.model_name)+"/tsne/tsne_" + str(epoch_idx) + ".png")
            plt.close(fig)
        log_data = {
            "val/train_class_acc": train_accuracy,
            "val/valid_class_acc": valid_accuracy,
        }
        print(f"\n Epoch {epoch_idx}: Train acc: {train_accuracy:.2f}%, Valid acc: {valid_accuracy:.2f}%")
        self.logger.log_metrics(log_data, step=epoch_idx)
    
    def run_model_init(self, exmp_input, init_rng):
        rng = jax.random.PRNGKey(self.seed)
        return self.forward.init(rng, exmp_input, is_training=True)


    





