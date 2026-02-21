from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    name: str
    train_path: str
    test_path: str = ""
    x_field: str = "a"
    y_field: str = "u"
    format: str = "mat"  # mat | npy | npz
    ntrain: int = 1000
    ntest: int = 100
    sub: int = 1
    resolution: List[int] = field(default_factory=lambda: [64])
    normalize_x: bool = False
    normalize_y: bool = False
    boundary_zero: bool = False


@dataclass
class ModelConfig:
    model_type: str = "vnngp"  # vnngp | svgp
    num_latents: int = 60
    num_nn: int = 50
    num_inducing: int = 100
    # Kernel: matern | rbf | spectral_mixture | rational_quadratic
    #         | periodic | gibbs | deep_kernel
    kernel: str = "matern"
    kernel_nu: float = 1.5
    num_mixtures: int = 4         # for spectral_mixture kernel
    use_ard: bool = False
    # Mean: constant | wno | nn | fourier_basis | polynomial_basis
    mean_type: str = "constant"
    nn_mean_hidden: int = 128     # hidden dim for NN mean
    nn_mean_layers: int = 3       # hidden layers for NN mean
    fourier_modes: int = 8        # modes for Fourier basis mean
    poly_degree: int = 3          # degree for polynomial basis mean
    use_wno_embedding: bool = False
    wno_width: int = 64
    wno_level: int = 4
    wno_layers: int = 4
    wno_wavelet: str = "db4"
    wno_embed_dim: int = 32       # projection dim for WNO embedding


@dataclass
class TrainingConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 16
    num_epochs: int = 200
    scheduler_step: int = 50
    scheduler_gamma: float = 0.75
    # Optimizer: adamw | adam | ngd | lbfgs | muon | schedule_free
    optimizer: str = "adamw"
    lr_ngd: float = 0.1           # NGD learning rate (only for 'ngd')
    muon_momentum: float = 0.95   # momentum (only for 'muon')
    warmup_steps: int = 0         # warmup (only for 'schedule_free')
    seed: int = 0


@dataclass
class LoggingConfig:
    use_wandb: bool = False
    wandb_project: str = "sgpo"
    wandb_run_name: str = ""
    save_dir: str = "results"
    save_model: bool = True
    save_predictions: bool = True
    log_interval: int = 10


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    device: str = "auto"
