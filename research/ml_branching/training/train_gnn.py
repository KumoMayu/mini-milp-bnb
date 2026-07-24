from __future__ import annotations

import argparse

from dataclasses import replace

from research.ml_branching.training.gnn_trainer import append_training_summary, load_gnn_training_config, train_gnn_one_config


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Train a bipartite GNN branching policy.")
    parser.add_argument("--config", required=True, help="Path to a GNN training config JSON file.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--round-max", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    config = load_gnn_training_config(args.config)
    updates = {}
    if args.run_name is not None:
        updates["run_name"] = args.run_name
    if args.round_max is not None:
        updates["round_max"] = args.round_max
    if args.max_epochs is not None:
        updates["max_epochs"] = args.max_epochs
    if args.seed is not None:
        updates["seed"] = args.seed
    if updates:
        config = replace(config, **updates)
    row = train_gnn_one_config(config)
    append_training_summary(row, config.report_dir)
    print(
        "trained_gnn "
        f"run={row['run_name']} checkpoint={row['checkpoint_path']} "
        f"val_regret={float(row['validation_normalized_regret']):.6g} "
        f"val_top1={float(row['validation_top1_accuracy']):.3f}"
    )
    return row


if __name__ == "__main__":
    main()
