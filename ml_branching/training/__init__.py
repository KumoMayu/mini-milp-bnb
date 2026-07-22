from .config import TrainingConfig, load_training_config

__all__ = ["TrainingConfig", "load_training_config", "train_one_config", "run_training_grid"]


def __getattr__(name: str):
    if name in {"run_training_grid", "train_one_config"}:
        from .trainer import run_training_grid, train_one_config

        return {"run_training_grid": run_training_grid, "train_one_config": train_one_config}[name]
    raise AttributeError(name)
