from functools import partial
from typing import List
from typing import Optional
from dataclasses import dataclass
import attr
from src.model.vicreg import VICReg

@dataclass
@attr.s(auto_attribs=True)
class ModelParams:
    # encoder model selection
    #model_name:"VICReg"
    #model_class:VICReg
    encoder_arch: str = "resnet18"
    shuffle_batch_norm: bool = False
    embedding_dim: int = 512  # must match embedding dim of encoder
    w: float = 0.2
    beta: float = 0.0
    # data-related parameters
    dataset_name: str = "cifar10"
    batch_size: int = 512
    ema_preset: dict = {40: 0.97, 100: 0.99, 300: 0.99, 1000: 0.996}

    # MoCo parameters
    dim: int = 128

    # optimization parameters
    lr: float = 0.0009
    momentum: float = 0.9
    weight_decay: float = 0.04#1e-5 #5e-4
    max_epochs: int = 300
    final_lr_schedule_value: float = 0.0

    # transform parameters
    transform_s: float = 0.5
    transform_apply_blur: bool = True

    # Change these to make more like BYOL
    use_momentum_schedule: bool = False
    loss_type: str = "ce"
    use_negative_examples_from_queue: bool = True
    use_both_augmentations_as_queries: bool = False
    optimizer_name: str = "sgd"
    lars_warmup_epochs: int = 1
    lars_eta: float = 1e-3
    exclude_matching_parameters_from_lars: List[str] = []  # set to [".bias", ".bn"] to match paper
    loss_constant_factor: float = 1

    # Change these to make more like VICReg
    use_vicreg_loss: bool = False
    use_lagging_model: bool = True
    use_unit_sphere_projection: bool = True
    invariance_loss_weight: float = 25.0
    variance_loss_weight: float = 25.0
    covariance_loss_weight: float = 1.0
    variance_loss_epsilon: float = 1e-04

    # MLP parameters
    projection_mlp_layers: int = 2
    prediction_mlp_layers: int = 0
    mlp_hidden_dim: int = 512

    mlp_normalization: Optional[str] = None
    prediction_mlp_normalization: Optional[str] = "same"  # if same will use mlp_normalization
    use_mlp_weight_standardization: bool = False

    # data loader parameters
    num_data_workers: int = 8
    drop_last_batch: bool = True
    pin_data_memory: bool = True
    gather_keys_for_queue: bool = False

    #Checkpoints
    log_every_n_steps: int = 1

    #Mean Shift parameters
    mem_bank_size: int = 128000
    top_k: int = 5

VICRegParams = partial(
    ModelParams,
    #model_name="VICReg",
    #model_class=VICReg,
    optimizer_name="adam",
    exclude_matching_parameters_from_lars=[],#[".bias", ".bn"],
    final_lr_schedule_value=0.002,
    mlp_normalization="bn",
    lars_warmup_epochs=10,
)