from config.model_params import VICRegParams
from src.train.trainer import Trainer
from src.dataloader.augment import numpy_collate_contrastive, numpy_collate
from src.dataloader.augment import ContrastiveTransformations, parallel_augment, contrast_transforms, TestTransformations, test_transforms, test_augment
from torchvision.datasets import CIFAR10
import torch
import torch.utils.data as data
import attr
import jax

DATASET_PATH = "../data"

model_params = attr.asdict(VICRegParams())
n_patches = 2
unlabeled_data = CIFAR10(root=DATASET_PATH, train=True, download=True, transform=ContrastiveTransformations(contrast_transforms, n_views=n_patches))
train_data_contrast = CIFAR10(root=DATASET_PATH, train=False, download=True, transform=ContrastiveTransformations(test_transforms, n_views=n_patches))
train_data_test = CIFAR10(root=DATASET_PATH, train=False, download=True, transform=TestTransformations(test_transforms, n_views=1))


train_loader = data.DataLoader(unlabeled_data,
                            batch_size=model_params['batch_size'],
                            shuffle=True,
                            drop_last=True,
                            collate_fn=numpy_collate_contrastive,
                            num_workers=4,
                            #persistent_workers=True,
                            generator=torch.Generator().manual_seed(42))

val_loader = data.DataLoader(train_data_contrast,
                            batch_size=model_params['batch_size'],
                            shuffle=False,
                            drop_last=True,
                            collate_fn=numpy_collate_contrastive,)

test_loader = data.DataLoader(train_data_test,
                            batch_size=1000,
                            shuffle=False,
                            drop_last=True,
                            collate_fn=numpy_collate)


def train(model_params, **kwargs):
    # Create a trainer module with specified hyperparameters
    trainer_instance = Trainer(exmp_input=parallel_augment(jax.random.PRNGKey(0),next(iter(train_loader))),
                    model_hparams={'n_patches': n_patches, },
                    optimizer_hparams={
                        'optimizer': model_params['optimizer_name'],
                        #'momentum': model_params['momentum'],
                        #'weight_decay': model_params['weight_decay'],
                        'lr': model_params['lr'],
                    },
                    #logger_params = {
                    #    'logger_type': 'wandb'
                    #},
                    debug = False,
                    **kwargs)

    trainer_instance.train_model(train_loader, val_loader, test_loader, num_epochs=model_params['max_epochs'])
    return trainer_instance

trainer_instance = train(model_params)