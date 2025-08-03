from dataclasses import dataclass

@dataclass
class Config:
    # Data
    image_size: int = 50
    channels: int = 3
    batch_size: int = 256
    buffer_size: int = 4096
    test_size: float = 0.2      # 80/20 split by default
    val_size: float = 0.1       # of train portion (so ~72/8/20)
    seed: int = 42

    # Training
    epochs: int = 15
    learning_rate: float = 1e-3
    patience_es: int = 4
    patience_rlr: int = 2

    # Paths
    models_dir: str = "models"
    outputs_dir: str = "outputs"
    log_dir: str = "outputs/tensorboard"
    best_model_path: str = "models/cancernet_best.h5"
