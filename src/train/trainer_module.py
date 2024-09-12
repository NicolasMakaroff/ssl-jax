import contextlib
import functools

import os
from typing import Iterable, Mapping, NamedTuple, Tuple, Any, Dict, Optional, Callable, Iterator
import json
import time 
from tqdm.auto import tqdm
from copy import copy
from collections import defaultdict

from absl import app
from absl import flags
from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import jmp
import optax
import pickle

from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from src.model.simsiam import l2_normalize

class TrainState(NamedTuple):
    params: hk.Params
    target_params: hk.Params
    state: hk.State
    target_state: hk.State
    opt_state: optax.OptState
    loss_scale: jmp.LossScale
    rng: Any
    queue: jnp.array
    queue_ptr: int

class TrainerModule:

    def __init__(self,
                model_name: str,
                model_class: hk.Module,
                model_hparams: Dict[str, Any],
                optimizer_hparams: Dict[str, Any],
                #eval_key: str,
                exmp_input: Any,
                precision_policy: str = 'p=f32,c=f32,o=f32',
                precision_bn_policy: str = 'p=f32,c=f32,o=f32',
                seed: int = 42,
                logger_params: Dict[str, Any] = None,
                enable_progress_bar: bool = True,
                debug: bool = False,
                pretrained: bool = False,
                check_val_every_n_epoch: int = 1,
                base_target_ema: float = None,
                is_byol: bool = False,
                max_steps: int = 50000,
                hparams: Dict[str, Any] = None,
                **kwargs):
        """
        A basic Trainer module summarizing most common training functionalities
        like logging, model initialization, training loop, etc.

        Atributes:
          model_class: The class of the model that should be trained.
          model_hparams: A dictionary of all hyperparameters of the model. Is
            used as input to the model when created.
          optimizer_hparams: A dictionary of all hyperparameters of the optimizer.
            Used during initialization of the optimizer.
          exmp_input: Input to the model for initialization and tabulate.
          seed: Seed to initialize PRNG.
          logger_params: A dictionary containing the specification of the logger.
          enable_progress_bar: If False, no progress bar is shown.
          debug: If True, no jitting is applied. Can be helpful for debugging.
          check_val_every_n_epoch: The frequency with which the model is evaluated
            on the validation set.
        """
        super().__init__()
        self.model_name = model_name
        self.model_class = model_class
        self.model_hparams = model_hparams
        self.optimizer_hparams = optimizer_hparams
        #self.eval_key = eval_key
        self.exmp_input = exmp_input
        self.precision_policy = lambda: jmp.get_policy(precision_policy)
        self.precision_bn_policy = lambda: jmp.get_policy(precision_bn_policy)
        self.seed = seed
        self.enable_progress_bar = enable_progress_bar
        self.debug = debug
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.optimizer = None
        self.pretrained = pretrained
        self._base_target_ema = base_target_ema
        self.is_byol = is_byol
        self._max_steps = max_steps
        self.hparams = hparams
        # Set of hyperparameters to save
        self.config = {
            'model_class': model_class.__name__,
            'model_hparams': model_hparams,
            'optimizer_hparams': optimizer_hparams,
            'logger_params': logger_params,
            'precision_policy': precision_policy,
            'precision_bn_policy': precision_bn_policy,
            'enable_progress_bar': self.enable_progress_bar,
            'debug': self.debug,
            'check_val_every_n_epoch': check_val_every_n_epoch,
            'seed': self.seed,
            'pretrained': self.pretrained,
        }
        self.config.update(kwargs)
        # Create empty model
        self.forward = hk.transform_with_state(self._forward)
        
        # Init trainer parts
        self.init_logger(logger_params)
        self.create_jitted_functions()
        if pretrained: self.load_model(name='checkpoints/' + self.model_name, step=5)
        else: self.init_model(exmp_input)
        #self.print_tabulate(exmp_input, self.train_step)

    def _forward(self,
                batch,
                is_training: bool = True) -> jnp.array:
        net = self.model_class(**self.model_hparams)
        if is_training: return net(batch, is_training=is_training)
        else: return net.encode(batch, is_training=is_training)

    def get_initial_loss_scale(self, mp_scale_value: float = 2 ** 15, mp_scale_type: str = 'NoOp') -> jmp.LossScale:
            """
            Returns the initial loss scale for mixed precision training.

            Args:
                mp_scale_value (float): The initial loss scale value. Default is 2 ** 15.
                mp_scale_type (str): The type of loss scale to use. Options are 'NoOp', 'Static', 'Dynamic'. Default is 'NoOp'.

            Returns:
                jmp.LossScale: The initial loss scale object.

            Raises:
                AttributeError: If the specified loss scale type is not supported.
            """
            cls = getattr(jmp, f'{mp_scale_type}LossScale')
            return cls(mp_scale_value) if cls is not jmp.NoOpLossScale else cls()


    def init_logger(self,
                    logger_params: Optional[Dict] = None):
        
        def init_logger(self, logger_params: Optional[Dict] = None):
            """
            Initializes a logger and creates a logging directory.

            Args: 
                logger_params (Optional[Dict]): A dictionary with the specifications of the logger.

            The method first checks if `logger_params` is None. If it is, an empty dictionary is assigned to `logger_params`.
            Then, it determines the logging directory based on the value of `base_log_dir` in `logger_params`. If `base_log_dir` is not provided, the default value is 'checkpoints/'.
            The logging directory is created by joining `base_log_dir`, the model class name, and the logger name (if provided) using `os.path.join()`.
            The version is set to None if `log_dir` is not provided, otherwise it is set to an empty string.
            The logger type is determined based on the value of `logger_type` in `logger_params`. If it is 'TensorBoard', a `TensorBoardLogger` object is created with the specified save directory, version, and name. If it is 'wandb', a `WandbLogger` object is created with the specified project name, save directory, version, and configuration.
            If an unknown logger type is provided, an assertion error is raised.
            The hyperparameters are saved in the logging directory as 'hparams.json' if the file does not already exist.
            The logging directory is assigned to `self.log_dir`.

            """
        if logger_params is None:
            logger_params = dict()
        # Determine logging directory
        log_dir = logger_params.get('base_log_dir', 'checkpoints/')
        if not log_dir:
            base_log_dir = logger_params.get('base_log_dir', 'checkpoints/')
            # Prepare logging
            log_dir = os.path.join(base_log_dir, self.config["model_class"])
            if 'logger_name' in logger_params:
                log_dir = os.path.join(log_dir, logger_params['logger_name'])
            version = None
        else:
            version = ''
        # Create logger object
        logger_type = logger_params.get('logger_type', 'TensorBoard').lower()
        if logger_type == 'tensorboard':
            self.logger = TensorBoardLogger(save_dir=log_dir,
                                            version=version,
                                            name='')
        elif logger_type == 'wandb':
            self.logger = WandbLogger(name=logger_params.get('project_name', None),
                                    save_dir=log_dir,
                                    version=version,
                                    config=self.config)
        else:
            assert False, f'Unknown logger type\"{logger_type}"'
        # Save hyperparameters
        log_dir = self.logger.save_dir
        if not os.path.isfile(os.path.join(log_dir, 'hparams.json')):
            os.makedirs(os.path.join(log_dir, 'metrics/'), exist_ok=True)
            with open(os.path.join(log_dir, 'hparams.json'), 'w') as f:
                json.dump(self.config, f, indent=4)
        self.log_dir = log_dir

    def init_model(self,
                exmp_input: Any,):
            """
            Creates an initial training state with newly generated network parameters.

            Args:
                exmp_input: An input to the model with which the shapes are inferred.

            Returns:
                None
            """
            # For initialization we need the same random key on each device.
            #local_device_count = jax.local_device_count()
            rng = jax.random.PRNGKey(self.seed)
            #rng = jnp.broadcast_to(rng, (local_device_count,) + rng.shape)
            #exmp_input = [exmp_input] if not isinstance(exmp_input, (list, tuple)) else exmp_input
            # Run model initialization
            if not self.is_byol:
                params, state = self.run_model_init(exmp_input, rng)
                loss_scale = self.get_initial_loss_scale()
                # Create default state. Optimizer is initialized later
                self.state = TrainState(params=params,
                                        target_params=None,
                                        state=state,
                                        target_state=None,
                                        opt_state=None,
                                        loss_scale=loss_scale,
                                        rng=rng,
                                        queue=None,
                                        queue_ptr=None)
            else: 
                rng_online, rng_target = jax.random.split(rng)
                online_params, online_state = self.run_model_init(exmp_input, rng_online,)
                target_params, target_state = self.run_model_init(exmp_input, rng_target,)
                loss_scale = self.get_initial_loss_scale()
                # Create default state. Optimizer is initialized later
                self.state = TrainState(params=online_params,
                                        target_params=target_params,
                                        state=online_state,
                                        target_state=target_state,
                                        opt_state=None,
                                        loss_scale=loss_scale,
                                        rng=rng,
                                        queue=l2_normalize(jax.random.normal(jax.random.PRNGKey(0), (self.hparams['mem_bank_size'], self.hparams['embedding_dim']))),
                                        queue_ptr = 0,)
             

    def run_model_init(self,
                    exmp_input: Any,
                    init_rng: Any,) :
        """
        The model initialization call.
        TODO Find a way to add `jax.pmap`to perform initialization (ex: jax.pmap(initial_state)(rng, batch)) related to batch creation

        Args:
          exmp_input: An input to the model with which the shapes are inferred.
          init_rng: A jax.random.PRNGKey.

        Returns:
          The initialized variable dictionary.
        """
        return self.forward.init(init_rng, exmp_input, is_training=True)

    def print_tabulate(self,
                        batch,
                        train_step):
            """
            Print a useful summary of the execution of our module.

            Args:
                train_step: The train step function that updates the module's state.
                batch: The input batch for the train step function.
            """
            summary = hk.experimental.tabulate(train_step)(self.state, batch)
            for line in summary.split('\n'):
                logging.info(line)

    def init_optimizer(self,
                        num_epochs: int,
                        num_steps_per_epoch: int,):
            """
            Initializes the optimizer for training.

            Args:
                num_epochs (int): The number of epochs for training.
                num_steps_per_epoch (int): The number of steps per epoch.

            Returns:
                optimizer: The initialized optimizer.
            """
            hparams = copy(self.optimizer_hparams)

            # Initialize optimizer
            optimizer_name = hparams.pop('optimizer','sgd')
            if optimizer_name.lower() == 'adam':
                opt_class = optax.adam
            elif optimizer_name.lower() == 'adamw':
                opt_class = optax.adamw
            elif optimizer_name.lower() == 'sgd':
                opt_class = optax.sgd
            else:
                assert False, f'Unknown optimizer "{optimizer_name}"'
            # Initialize learning rate scheduler 
            # A cosine decay scheduler is used 
            lr = hparams.pop('lr', 1e-3)
            warmup = hparams.pop('warmup', 0)
            lr_schedule = optax.warmup_cosine_decay_schedule(
                init_value=lr,
                peak_value=lr,
                warmup_steps=warmup,
                decay_steps=int(num_epochs * num_steps_per_epoch),
                end_value=0.0 * lr
            )
            # Clip gradient at max value and evt apply weight decay
            transf = [optax.clip_by_global_norm(hparams.pop('gradient_clip', 1.0))]
            if opt_class == optax.sgd and 'weight_decay' in hparams: # wd is integrated in adamw
                transf.append(optax.add_decayed_weights(hparams.pop('weight_decay', 0.0)))
            optimizer = lambda: optax.chain(
                *transf,
                opt_class(lr_schedule, **hparams)
            )   
            # Initialize training state
            self.state = TrainState(params=self.state.params,
                                    target_params=self.state.target_params,
                                    state=self.state.state,
                                    target_state=self.state.target_state,
                                    opt_state=optimizer().init(self.state.params,),
                                    loss_scale=self.state.loss_scale,
                                    rng=self.state.rng,
                                    queue =self.state.queue,
                                    queue_ptr=self.state.queue_ptr)
           
            return optimizer

    def create_jitted_functions(self):
        """
        Creates jitted versions of the training and evaluation functions.
        If self.debug is True, no jitting is applied.
        """
        train_step, eval_step = self.create_functions()
        if self.debug:
            print('Skipping jitting due to debug=True\n')
            self.train_step = train_step
            self.eval_step = eval_step
        else:
            self.train_step = jax.jit(train_step)
            self.eval_step = jax.jit(eval_step)
        

    def create_functions(self) -> Tuple[Callable[[TrainState, Any], Tuple[TrainState, Dict]],
                                        Callable[[TrainState, Any], Tuple[TrainState, Dict]]]:

        """
        Creates and returns functions for the training and evaluation step. The
        functions take as input the training state and a batch from the train/
        val/test loader. Both functions are expected to return a dictionary of
        logging metrics, and the training function a new train state. This
        function needs to be overwritten by a subclass. The train_step and
        eval_step functions here are examples for the signature of the functions.
        """
        def train_step(state : TrainState,
                       batch : Any,
                       epoch_idx: int):
            metrics = {}
            return state, metrics
        def eval_step(state : TrainState,
                      batch : Any):
            metrics = {}
            return metrics
        raise NotImplementedError

    def train_model(self,
                    train_loader: Iterator,
                    val_loader: Iterator,
                    test_loader: Optional[Iterator] = None,
                    num_epochs: int = 100) -> Dict[str, Any]:

        """
        Starts a training loop for the given number of epochs.

        Args:
          train_loader: Data loader of the training set.
          val_loader: Data loader of the validation set.
          test_loader: If given, best model will be evaluated on the test set.
          num_epochs: Number of epochs for which to train the model.

        Returns:
          A dictionary of the train, validation and evt. test metrics for the
          best model on the validation set.
        """
        # Create optimizer and the scheduler for the given number of epochs
        self.optimizer = self.init_optimizer(num_epochs, len(train_loader))
        # Prepare training loop
        self.on_training_start()
        best_eval_metrics = None
        # Initial evaluation
        self.on_validation_epoch_end(0, None, test_loader)
        tracker = self.tracker(range(1,num_epochs+1), desc='Epochs')
        for epoch_idx in tracker:
            train_metrics = self.train_epoch(train_loader, epoch_idx)
            tracker.set_postfix(loss=f"{train_metrics['train/loss']:.4f}")
            self.logger.log_metrics(train_metrics, step=epoch_idx)
            self.on_training_epoch_end(epoch_idx)
            # Validation every N epochs
            if epoch_idx % self.check_val_every_n_epoch == 0:
                eval_metrics = self.eval_model(val_loader, log_prefix='val/')
                self.on_validation_epoch_end(epoch_idx, eval_metrics, test_loader)
                self.logger.log_metrics(eval_metrics, step=epoch_idx)
                self.save_metrics(f'eval_epoch_{str(epoch_idx).zfill(3)}', eval_metrics)
                # Save best model
                if self.is_new_model_better(eval_metrics, best_eval_metrics):
                    best_eval_metrics = eval_metrics
                    best_eval_metrics.update(train_metrics)
                    #self.save_model(step=0, name=self.model_name, ckpt_dir='asset/' + str(self.model_name) + '/weights/')
                    self.save_metrics('best eval', eval_metrics)
        # Test best model if possible
        if test_loader is not None:
            self.load_model(name = 'asset/' + str(self.model_name) + '/weights/', step=0)
            test_metrics = self.eval_model(test_loader, log_prefix='test/')
            self.logger.log_metrics(test_metrics, step=epoch_idx)
            self.save_metrics('test', test_metrics)
            best_eval_metrics.update(test_metrics)
        # Close logger
        self.logger.finalize('success')
        return best_eval_metrics

    def train_epoch(self,
                    train_loader: Iterator,
                    epoch_idx: int) -> Dict[str, Any]:

        """
        Trains a model for one epoch.

        Args:
          train_loader: Data loader of the training set.

        Returns:
          A dictionary of the average training metrics over all batches
          for logging.
        """
        # Train model for one epoch, and log avg loss and accuracy
        metrics = defaultdict(float)
        num_train_steps = len(train_loader)
        start_time = time.time()
        tracker = self.tracker(train_loader, desc='Training', leave=False)
        for batch in tracker:
            self.state, step_metrics, _ = self.train_step(self.state, batch, epoch_idx)
            tracker.set_postfix(loss=f"{step_metrics['loss']:.4f}")
            for key in step_metrics:
                metrics['train/' + key] += step_metrics[key] / num_train_steps
        metrics = {key: metrics[key].item() for key in metrics}
        metrics['epoch_time'] = time.time() - start_time
        return metrics

    def eval_model(self,
                    data_loader: Iterator,
                    log_prefix: Optional[str] = '') -> Dict[str, Any]:
        
        """
        Evaluates the model on a dataset.

        Args:
          data_loader: Data loader of the dataset to evaluate on.
          log_prefix: Prefix to add to all metrics (e.g. 'val/' or 'test/')

        Returns:
          A dictionary of the evaluation metrics, averaged over data points
          in the dataset.
        """
        # Test model on all images of a data loader and return avg loss
        metrics = defaultdict(float)
        num_elements = 0
        tracker = self.tracker(data_loader, desc='Validation', leave=False)
        for batch in tracker:
            step_metrics = self.eval_step(self.state, batch)
            batch_size = batch[0].shape[0] if isinstance(batch, (list, tuple)) else batch.shape[0]
            for key in step_metrics:
                metrics[key] += step_metrics[key] * batch_size
            num_elements += batch_size
        metrics = {(log_prefix + key): (metrics[key] / num_elements).item() for key in metrics}
        return metrics

    def is_new_model_better(self,
                    new_metrics: Dict[str, Any],
                    old_metrics: Dict[str, Any]) -> bool:

        """
        Compares two sets of evaluation metrics to decide whether the
        new model is better than the previous ones or not.

        Args:
          new_metrics: A dictionary of the evaluation metrics of the new model.
          old_metrics: A dictionary of the evaluation metrics of the previously
            best model, i.e. the one to compare to.

        Returns:
          True if the new model is better than the old one, and False otherwise.
        """
        if old_metrics is None:
            return True
        for key, is_larger in [('val/val_metric', False), ('val/acc', True), ('val/loss', False)]:
            if key in new_metrics:
                if is_larger:
                    return new_metrics[key] > old_metrics[key]
                else:
                    return new_metrics[key] < old_metrics[key]
        assert False, f'No known metrics to log on: {new_metrics}'

    def tracker(self,
                iterator: Iterator,
                **kwargs) -> Iterator:
        
        if self.enable_progress_bar:
            return tqdm(iterator, **kwargs)
        else:
            return iterator

    def save_metrics(self,
                    filename: str,
                    metrics: Dict[str, Any]):

        """
        Saves a dictionary of metrics to file. Can be used as a textual
        representation of the validation performance for checking in the terminal.

        Args:
          filename: Name of the metrics file without folders and postfix.
          metrics: A dictionary of metrics to save in the file.
        """
        with open(os.path.join(self.log_dir, f'metrics/{filename}.json'), 'w') as f:
            json.dump(metrics, f, indent=4)


    def on_training_start(self):
        """
        Method called before training is started. Can be used for additional
        initialization operations etc.
        """
        pass

    def on_training_epoch_end(self,
                            epoch_idx: int):
        """
        Method called at the end of each training epoch. Can be used for additional
        logging or similar.

        Args:
          epoch_idx: Index of the training epoch that has finished.
        """
        pass

    def on_validation_epoch_end(self,
                                epoch_idx: int,
                                eval_metrics: Dict[str, Any],
                                val_loader: Iterator):
        """
        Method called at the end of each validation epoch. Can be used for additional
        logging and evaluation.

        Args:
          epoch_idx: Index of the training epoch at which validation was performed.
          eval_metrics: A dictionary of the validation metrics. New metrics added to
            this dictionary will be logged as well.
          val_loader: Data loader of the validation set, to support additional
            evaluation.
        """
        pass

    def save_model(self,
                   name: str = 'model',
                    step: int = 0,
                    ckpt_dir: str = './checkpoints/'):
        with open(os.path.join(ckpt_dir, f"/weights_array_{str(step)}.npy"), "wb") as f:
            for x in jax.tree_util.tree_leaves(self.state):
                np.save(f, x, allow_pickle=False)

        tree_struct = jax.tree_util.tree_map(lambda t: 0, self.state)
        with open(os.path.join(ckpt_dir, f"/weights_tree_{str(step)}.pkl"), "wb") as f:
            pickle.dump(tree_struct, f)

    def load_model(self, name:str, step: int=30):
        state = self.load_from_checkpoint(name=name, step=step)
        self.state = state

    def bind_model(self):
        # TODO implement bind_model
        pass

    def load_from_checkpoint(self,
                            name: str,
                            ckpt_dir: str = './checkpoints/',
                            step: int = 30,
                            ) -> Any:
        with open(os.path.join(ckpt_dir, f"{name}/weights_tree_{str(step)}.pkl"), "rb") as f:
            tree_struct = pickle.load(f)

            leaves, treedef = jax.tree_util.tree_flatten(tree_struct)
        with open(os.path.join(ckpt_dir, f"{name}/weights_array_{str(step)}.npy"), "rb") as f:
            flat_state = [np.load(f) for _ in leaves]

        return jax.tree_util.tree_unflatten(treedef, flat_state)
