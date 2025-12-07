# Fashion Item Generation

A reproducible implementation of fashion item generation using Deep Convolutional Generative Adversarial Networks (DCGAN) trained on Fashion-MNIST dataset.

## Features

- **Modern Architecture**: DCGAN with spectral normalization, gradient penalty, and exponential moving average (EMA)
- **Comprehensive Evaluation**: FID, Inception Score, Precision/Recall, and LPIPS diversity metrics
- **Reproducible Training**: Deterministic seeding, configuration management, and checkpointing
- **Interactive Demo**: Streamlit-based web interface for sample generation
- **Production Ready**: Type hints, comprehensive testing, and clean code structure

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Fashion-Item-Generation.git
cd Fashion-Item-Generation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

### Training

Train a new model with default configuration:

```bash
python scripts/train.py
```

Train with custom parameters:

```bash
python scripts/train.py --config configs/config.yaml training.max_epochs=100 training.batch_size=128
```

Resume training from checkpoint:

```bash
python scripts/train.py --resume checkpoints/epoch_050-fid_45.23.ckpt
```

### Sampling

Generate samples from a trained model:

```bash
python scripts/sample.py --checkpoint checkpoints/best_model.ckpt --num_samples 64
```

Generate interpolation between two random samples:

```bash
python scripts/sample.py --checkpoint checkpoints/best_model.ckpt --interpolate --interpolation_steps 10
```

### Interactive Demo

Launch the Streamlit demo:

```bash
streamlit run demo/streamlit_app.py
```

## Project Structure

```
fashion_item_generation/
├── src/fashion_generation/          # Main source code
│   ├── models/                     # Model implementations
│   │   ├── dcgan.py               # DCGAN model
│   │   └── __init__.py
│   ├── data/                      # Data loading and preprocessing
│   │   ├── fashion_mnist.py       # Fashion-MNIST data module
│   │   └── __init__.py
│   ├── training/                  # Training utilities
│   │   ├── trainer.py            # PyTorch Lightning trainer
│   │   ├── losses.py             # Loss functions
│   │   └── __init__.py
│   ├── evaluation/                # Evaluation metrics
│   │   ├── metrics.py             # FID, IS, Precision/Recall, LPIPS
│   │   └── __init__.py
│   ├── sampling/                  # Sampling utilities
│   │   ├── sampler.py            # Sample generation and visualization
│   │   └── __init__.py
│   ├── utils.py                   # Utility functions
│   └── __init__.py
├── configs/                       # Configuration files
│   ├── config.yaml               # Main configuration
│   ├── model/dcgan.yaml          # Model configuration
│   ├── data/fashion_mnist.yaml   # Data configuration
│   ├── training/basic.yaml       # Training configuration
│   └── evaluation/standard.yaml  # Evaluation configuration
├── scripts/                       # Training and sampling scripts
│   ├── train.py                  # Main training script
│   └── sample.py                 # Sampling script
├── demo/                          # Interactive demos
│   └── streamlit_app.py         # Streamlit web app
├── tests/                         # Unit tests
│   └── test_models.py           # Model tests
├── notebooks/                     # Jupyter notebooks
├── assets/                        # Generated samples and visualizations
├── checkpoints/                   # Model checkpoints
├── logs/                          # Training logs
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore file
└── README.md                     # This file
```

## Configuration

The project uses OmegaConf for configuration management. Main configuration files:

- `configs/config.yaml`: Main configuration with project settings
- `configs/model/dcgan.yaml`: Model architecture parameters
- `configs/data/fashion_mnist.yaml`: Data loading and preprocessing settings
- `configs/training/basic.yaml`: Training hyperparameters and optimization
- `configs/evaluation/standard.yaml`: Evaluation metrics configuration

### Key Configuration Options

**Model Configuration:**
- `z_dim`: Latent vector dimension (default: 100)
- `img_channels`: Number of image channels (default: 1 for grayscale)
- `img_size`: Image size (default: 28 for Fashion-MNIST)
- `use_spectral_norm`: Enable spectral normalization (default: true)
- `ema.enabled`: Enable exponential moving average (default: true)

**Training Configuration:**
- `max_epochs`: Maximum training epochs (default: 100)
- `batch_size`: Training batch size (default: 64)
- `optimizer`: Optimizer settings (Adam with lr=0.0002)
- `loss`: Loss function configuration (BCE, hinge, or Wasserstein)

**Evaluation Configuration:**
- `metrics`: List of metrics to compute (FID, IS, Precision/Recall, LPIPS)
- `eval_every_n_epochs`: Evaluation frequency (default: 5)
- `eval_samples`: Number of samples for evaluation (default: 1000)

## Model Architecture

### DCGAN Generator
- Input: 100-dimensional latent vector
- Architecture: Fully connected layers (256 → 512 → 1024 → 784)
- Activation: ReLU + Tanh output
- Normalization: Batch normalization (optional)
- Regularization: Spectral normalization (optional)

### DCGAN Discriminator
- Input: 28×28 grayscale images
- Architecture: Convolutional layers (64 → 128) + Classifier (1024 → 1)
- Activation: LeakyReLU + Sigmoid output
- Normalization: Batch normalization (optional)
- Regularization: Spectral normalization (optional)

### Training Features
- **Spectral Normalization**: Stabilizes training by constraining weight matrices
- **Gradient Penalty**: Optional regularization for Wasserstein GAN training
- **Exponential Moving Average**: Improves generation quality by averaging model parameters
- **Mixed Precision**: Optional 16-bit training for memory efficiency

## Evaluation Metrics

### Fréchet Inception Distance (FID)
Measures the distance between real and generated image distributions in feature space. Lower is better.

### Inception Score (IS)
Evaluates both quality and diversity of generated images. Higher is better.

### Precision and Recall
- **Precision**: Measures quality of generated samples
- **Recall**: Measures diversity and coverage of the data distribution

### LPIPS Diversity
Measures perceptual diversity between generated samples using learned perceptual similarity.

## Dataset

The model is trained on the **Fashion-MNIST** dataset, which contains 70,000 grayscale images of 10 fashion categories:

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

Each image is 28×28 pixels and represents a fashion item from Zalando's website.

## Training

### Basic Training
```bash
python scripts/train.py
```

### Advanced Training Options
```bash
# Custom configuration
python scripts/train.py --config configs/config.yaml

# Override specific parameters
python scripts/train.py training.max_epochs=200 training.batch_size=128

# Resume from checkpoint
python scripts/train.py --resume checkpoints/epoch_050.ckpt

# Run evaluation only
python scripts/train.py --test

# Generate samples after training
python scripts/train.py --sample
```

### Training Monitoring
- **TensorBoard**: `tensorboard --logdir logs/tensorboard`
- **Weights & Biases**: Enable in configuration for cloud logging
- **Console**: Real-time loss monitoring and sample generation

## Sampling and Generation

### Command Line Sampling
```bash
# Generate sample grid
python scripts/sample.py --checkpoint checkpoints/best_model.ckpt --grid_size 8

# Generate specific number of samples
python scripts/sample.py --checkpoint checkpoints/best_model.ckpt --num_samples 100

# Generate with specific seed
python scripts/sample.py --checkpoint checkpoints/best_model.ckpt --seed 42

# Generate interpolation
python scripts/sample.py --checkpoint checkpoints/best_model.ckpt --interpolate
```

### Interactive Demo
Launch the Streamlit demo for interactive sample generation:

```bash
streamlit run demo/streamlit_app.py
```

Features:
- Real-time sample generation
- Adjustable parameters (number of samples, grid size, seed)
- Latent space interpolation
- Model information display

## Testing

Run unit tests:

```bash
pytest tests/
```

Run specific test file:

```bash
pytest tests/test_models.py
```

Run with coverage:

```bash
pytest --cov=src tests/
```

## Development

### Code Quality
The project uses:
- **Black**: Code formatting
- **Ruff**: Linting and import sorting
- **Pre-commit**: Git hooks for code quality

Setup pre-commit hooks:
```bash
pre-commit install
```

### Adding New Features
1. Create feature branch
2. Implement changes with tests
3. Update documentation
4. Run tests and linting
5. Submit pull request

## Performance

### Expected Results
On Fashion-MNIST dataset:
- **FID**: ~15-25 (lower is better)
- **Inception Score**: ~2.5-3.5 (higher is better)
- **Training Time**: ~2-4 hours on GPU (RTX 3080)
- **Memory Usage**: ~2-4 GB GPU memory

### Optimization Tips
- Use mixed precision training for memory efficiency
- Enable gradient accumulation for larger effective batch sizes
- Use spectral normalization for training stability
- Enable EMA for better generation quality

## Troubleshooting

### Common Issues

**CUDA Out of Memory:**
- Reduce batch size in configuration
- Enable gradient accumulation
- Use mixed precision training

**Training Instability:**
- Enable spectral normalization
- Reduce learning rate
- Use gradient clipping
- Enable EMA

**Poor Generation Quality:**
- Increase training epochs
- Enable EMA
- Adjust loss function (try hinge loss)
- Increase model capacity

### Getting Help
- Check the logs in `logs/` directory
- Review configuration files in `configs/`
- Run tests to verify installation
- Check GPU memory usage with `nvidia-smi`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Fashion-MNIST dataset by Zalando Research
- PyTorch Lightning for training framework
- Clean-FID for evaluation metrics
- Streamlit for interactive demo
# Fashion-Item-Generation
