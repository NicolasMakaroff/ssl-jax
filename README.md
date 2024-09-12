
# SSL Algorithm in JAX

## Description
This repository provides implementations of classical **Self-Supervised Learning (SSL)** algorithms for image classification using the [JAX](https://github.com/google/jax) library. The goal is to offer efficient, flexible, and reproducible implementations for SSL research and practical applications in unsupervised feature learning from images.

## Features
- **Self-Supervised Learning**: Implementation of popular SSL algorithms, such as:
  - SimSiam
  - BYOL (Bootstrap Your Own Latent)
  - VICReg
- **JAX Backend**: Utilizes JAX for efficient automatic differentiation, GPU/TPU acceleration, and vectorized operations.
- **Unsupervised Representation Learning**: Designed for learning useful image representations without labeled data.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ssl-algorithm-in-jax.git
   cd ssl-algorithm-in-jax
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. (Optional) Install JAX with CUDA support for GPU acceleration:
   ```bash
   pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   ```

## Usage

### Pretraining (SSL)

Train a model using one of the implemented SSL algorithms, such as SimSiam or BYOL:
```bash
python train.py 
```

### Configuration

You can modify various hyperparameters and configurations through the `config/model_params.py` file.

