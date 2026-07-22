from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_branching.training.data.collector import collect_instance_samples
from ml_branching.unit_commitment import generate_from_config


def collect_unit_commitment_dataset(config: dict, resume: bool = True) -> dict:
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and resume:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"dataset_id": config["dataset_id"], "instances": [], "samples": []}
    done = {row["instance_id"] for row in manifest.get("instances", [])}
    master_seed = int(config.get("master_seed", 20260715))
    for split, spec in dict(config["splits"]).items():
        for index in range(int(spec.get("instances", 1))):
            generated = generate_from_config(split, spec, index, master_seed)
            if generated.instance_id in done:
                continue
            result = collect_instance_samples(
                generated,
                out_dir=out_dir,
                dataset_id=config["dataset_id"],
                lp_backend="scipy_highs",
                max_nodes_per_instance=int(config.get("max_nodes_per_instance", 120)),
                max_decisions_per_instance=int(config.get("max_decisions_per_instance", 30)),
                time_limit_per_instance=float(config.get("time_limit_per_instance", 30.0)),
                use_matrix_presolve=True,
                round_id=0,
            )
            row = result.to_manifest_row()
            row["parameters"] = generated.parameters
            manifest.setdefault("instances", []).append(row)
            for sample_path in row.get("sample_paths", []):
                manifest.setdefault("samples", []).append(
                    {
                        "path": sample_path,
                        "split": split,
                        "family_name": "unit_commitment",
                        "instance_id": generated.instance_id,
                        "seed": generated.seed,
                        "units": generated.units,
                    }
                )
            done.add(generated.instance_id)
            print(f"accepted {generated.instance_id}: samples={result.samples_written} status={result.status}")
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    manifest["total_samples"] = len(manifest.get("samples", []))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Collect strong-branching expert data for unit commitment only.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    manifest = collect_unit_commitment_dataset(config, resume=not args.no_resume)
    print(json.dumps({"dataset_id": manifest["dataset_id"], "samples": manifest.get("total_samples", 0)}, indent=2))
    return manifest


if __name__ == "__main__":
    main()
