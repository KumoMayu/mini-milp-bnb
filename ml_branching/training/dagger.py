from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_branching.training.data.collector import collect_instance_samples
from ml_branching.runtime.inference import LearnedGNNBranchingPolicy
from ml_branching.unit_commitment import instance_from_parameters


def _load_manifest(dataset_path: str | Path) -> dict:
    return json.loads((Path(dataset_path) / "manifest.json").read_text(encoding="utf-8"))


def train_instance_parameters(dataset_path: str | Path) -> list[dict]:
    manifest = _load_manifest(dataset_path)
    params = []
    seen = set()
    for row in manifest.get("instances", []):
        parameters = row.get("parameters")
        if parameters is None and row.get("sample_paths"):
            sample_path = Path(row["sample_paths"][0])
            if not sample_path.is_absolute():
                sample_path = ROOT / sample_path
            with np.load(sample_path, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata_json"].item()))
            parameters = metadata["instance_parameters"]
        if not parameters or parameters.get("split") != "train":
            continue
        instance_id = str(parameters["instance_id"])
        if instance_id in seen:
            continue
        seen.add(instance_id)
        params.append(parameters)
    return params


def existing_round_instance_ids(dataset_path: str | Path, round_id: int) -> set[str]:
    manifest_path = Path(dataset_path) / f"dagger_round_{int(round_id)}_manifest.json"
    if not manifest_path.exists():
        return set()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {str(row["instance_id"]) for row in manifest.get("instances", []) if not row.get("skipped")}


def collect_dagger_round(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    round_id: int,
    max_instances: int | None = None,
    max_nodes_per_instance: int = 120,
    max_decisions_per_instance: int = 20,
    time_limit_per_instance: float = 30.0,
    device: str = "cpu",
    resume: bool = True,
) -> dict:
    dataset_path = Path(dataset_path)
    start = perf_counter()
    policy = LearnedGNNBranchingPolicy.from_checkpoint(checkpoint_path, device=device)
    all_params = train_instance_parameters(dataset_path)
    already_done = existing_round_instance_ids(dataset_path, round_id) if resume else set()
    rows = []
    samples = []
    selected = 0
    for parameters in all_params:
        if max_instances is not None and selected >= int(max_instances):
            break
        if str(parameters["instance_id"]) in already_done:
            continue
        generated = instance_from_parameters(parameters)
        result = collect_instance_samples(
            generated,
            out_dir=dataset_path,
            dataset_id=f"unit_commitment_gnn_dagger_round_{int(round_id)}",
            lp_backend="scipy_highs",
            max_nodes_per_instance=max_nodes_per_instance,
            max_decisions_per_instance=max_decisions_per_instance,
            time_limit_per_instance=time_limit_per_instance,
            use_matrix_presolve=True,
            control_policy=policy,
            control_policy_name=f"gnn_round_{int(round_id) - 1}",
            round_id=int(round_id),
        )
        row = result.to_manifest_row()
        rows.append(row)
        samples.extend(row.get("sample_paths", []))
        selected += 1

    manifest = {
        "round_id": int(round_id),
        "dataset_path": str(dataset_path),
        "checkpoint_path": str(checkpoint_path),
        "train_only": True,
        "instances": rows,
        "samples": samples,
        "runtime_sec": perf_counter() - start,
        "max_instances": max_instances,
        "max_nodes_per_instance": int(max_nodes_per_instance),
        "max_decisions_per_instance": int(max_decisions_per_instance),
        "time_limit_per_instance": float(time_limit_per_instance),
    }
    path = dataset_path / f"dagger_round_{int(round_id)}_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Collect train-only DAgger samples for the GNN branching policy.")
    parser.add_argument("--dataset", default="ml_branching/data/generated/unit_commitment_round0")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--round-id", type=int, required=True)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--max-nodes", type=int, default=120)
    parser.add_argument("--max-decisions", type=int, default=20)
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    manifest = collect_dagger_round(
        dataset_path=args.dataset,
        checkpoint_path=args.checkpoint,
        round_id=args.round_id,
        max_instances=args.max_instances,
        max_nodes_per_instance=args.max_nodes,
        max_decisions_per_instance=args.max_decisions,
        time_limit_per_instance=args.time_limit,
        device=args.device,
        resume=not args.no_resume,
    )
    print(
        json.dumps(
            {
                "round_id": manifest["round_id"],
                "instances": len(manifest["instances"]),
                "samples": len(manifest["samples"]),
                "runtime_sec": manifest["runtime_sec"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return manifest


if __name__ == "__main__":
    main()
