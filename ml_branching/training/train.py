from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_branching.training import load_training_config, run_training_grid


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Train a learned MLP branching scorer.")
    parser.add_argument("--config", default="ml_branching/configs/train_mlp.json")
    parser.add_argument("--data-dir", default=None, help="Compatibility override for dataset_path.")
    parser.add_argument("--output", default=None, help="Compatibility override for output_dir/checkpoint directory.")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    config = load_training_config(args.config)
    overrides = {}
    if args.data_dir is not None:
        overrides["dataset_path"] = args.data_dir
    if args.output is not None:
        overrides["output_dir"] = str(Path(args.output).parent if Path(args.output).suffix else Path(args.output))
    if args.max_epochs is not None:
        overrides["max_epochs"] = args.max_epochs
    if args.early_stopping_patience is not None:
        overrides["early_stopping_patience"] = args.early_stopping_patience
    if args.hidden_dim is not None:
        overrides["hidden_dim"] = args.hidden_dim
        overrides["hyperparameter_grid"] = None
    if args.seed is not None:
        overrides["seed"] = args.seed
        overrides["hyperparameter_grid"] = None
    if overrides:
        from dataclasses import replace

        config = replace(config, **overrides)
    summary = run_training_grid(config)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
