from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_branching.training.data.dataset import BranchingDataset
from ml_branching.training.data.schema import save_sample
from ml_branching.families import FAMILY_REGISTRY, FamilyInstance, get_family
from ml_branching.training.oracle.strong_branching import StrongBranchingPolicy, make_strong_branch_child_bounds
from solver.branch_and_bound import BBNode, check_feasibility, fractional_binary_candidates, is_binary_solution
from solver.branching import BranchingContext
from solver.lp_backends import get_lp_relaxation_solver
from solver.search_strategy import BestBoundNodePool


DEFAULT_CONFIG = {
    "dataset_id": "learning_branching_multifamily",
    "output_dir": "ml_branching/data/generated/smoke",
    "master_seed": 20260714,
    "splits": {
        "train": {"instances_per_family": 2, "sizes": [4, 5]},
        "validation": {"instances_per_family": 1, "sizes": [4, 5]},
        "in_distribution_test": {"instances_per_family": 1, "sizes": [4, 5]},
    },
    "families": sorted(FAMILY_REGISTRY),
    "max_nodes_per_instance": 80,
    "max_decisions_per_instance": 20,
    "time_limit_per_instance": 30.0,
    "max_generation_attempts": 200,
    "lp_backend": "scipy_highs",
    "scoring_mode": "product",
    "epsilon": 1e-6,
    "mu": 0.25,
    "max_lp_candidates": None,
    "use_matrix_presolve": True,
    "tolerance": 1e-8,
}


@dataclass
class InstanceCollectionResult:
    instance_id: str
    split: str
    family_name: str
    scale_group: str
    seed: int
    units: int
    status: str
    samples_written: int
    formal_lp_count: int
    formal_nodes_processed: int
    probe_lp_count: int
    probe_runtime_sec: float
    runtime_sec: float
    skipped: bool = False
    skip_reason: str = ""
    limit_reason: str = ""
    sample_paths: list[str] | None = None
    sample_summaries: list[dict] | None = None

    def to_manifest_row(self) -> dict:
        row = asdict(self)
        row["sample_paths"] = self.sample_paths or []
        row["sample_summaries"] = self.sample_summaries or []
        return row


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _safe_bound(value: float | None) -> float:
    return np.nan if value is None else float(value)


def _solve_child_lp(lp_solver, problem, child: BBNode, tolerance: float, max_lp_candidates, use_matrix_presolve: bool):
    return lp_solver(problem, child.lb, child.ub, tolerance, max_lp_candidates, use_matrix_presolve)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _sample_arrays(problem, node: BBNode, lp, records) -> dict:
    return {
        "internal_objective_coefficients": problem.internal_c,
        "G": problem.G,
        "h": problem.h,
        "node_lb": node.lb,
        "node_ub": node.ub,
        "lp_solution": lp.x,
        "binary_variable_indices": np.array(problem.binary_indices, dtype=int),
        "candidate_indices": np.array([r.candidate_index for r in records], dtype=int),
        "candidate_lp_values": np.array([r.candidate_lp_value for r in records], dtype=float),
        "child_0_bound": np.array([_safe_bound(r.child_0_bound) for r in records], dtype=float),
        "child_1_bound": np.array([_safe_bound(r.child_1_bound) for r in records], dtype=float),
        "child_0_runtime_sec": np.array([r.child_0_time_sec for r in records], dtype=float),
        "child_1_runtime_sec": np.array([r.child_1_time_sec for r in records], dtype=float),
        "delta_0": np.array([r.delta_0 for r in records], dtype=float),
        "delta_1": np.array([r.delta_1 for r in records], dtype=float),
        "expert_scores": np.array([r.score for r in records], dtype=float),
        "child0_bound": np.array([_safe_bound(r.child_0_bound) for r in records], dtype=float),
        "child1_bound": np.array([_safe_bound(r.child_1_bound) for r in records], dtype=float),
    }


def _sample_metadata(
    dataset_id: str,
    split: str,
    generated: FamilyInstance,
    node: BBNode,
    lp,
    node_bound: float,
    incumbent_value: float | None,
    records,
    expert_choice: int,
    scoring_mode: str,
    epsilon: float,
    mu: float,
    tolerance: float,
    lp_backend: str,
    solver_config: dict,
    git_commit: str | None,
    probe_lp_count: int,
    probe_runtime_sec: float,
    round_id: int = 0,
    control_policy_name: str = "strong_branching",
    control_branching_variable: int | None = None,
) -> dict:
    child_0_status = [r.child_0_status for r in records]
    child_1_status = [r.child_1_status for r in records]
    scores = [float(r.score) for r in records]
    metadata = {
        "dataset_id": dataset_id,
        "split": split,
        "family_name": generated.parameters.get("family_name", getattr(generated, "family_name", "unknown")),
        "scale_group": generated.parameters.get("scale_group", getattr(generated, "scale_group", "unknown")),
        "instance_id": generated.instance_id,
        "instance_seed": int(generated.seed),
        "instance_parameters": generated.parameters,
        "node_id": int(node.node_id),
        "node_depth": int(node.depth),
        "objective_sense": generated.problem.sense,
        "parent_internal_lp_bound": float(node_bound),
        "incumbent_internal_value": None if incumbent_value is None else float(incumbent_value),
        "child_0_status": child_0_status,
        "child_1_status": child_1_status,
        "child_0_runtime_sec": [float(r.child_0_time_sec) for r in records],
        "child_1_runtime_sec": [float(r.child_1_time_sec) for r in records],
        "expert_selected_variable": int(expert_choice),
        "scoring_mode": scoring_mode,
        "epsilon": float(epsilon),
        "mu": float(mu),
        "tolerance": float(tolerance),
        "lp_backend": lp_backend,
        "solver_config": solver_config,
        "probe_lp_count": int(probe_lp_count),
        "probe_runtime_sec": float(probe_runtime_sec),
        "generated_at": _now_iso(),
        "git_commit": git_commit,
        "config": generated.parameters,
        "round_id": int(round_id),
        "control_policy_name": str(control_policy_name),
        "control_branching_variable": None if control_branching_variable is None else int(control_branching_variable),
        "sense": generated.problem.sense,
        "parent_lp_bound": float(node_bound),
        "child0_status": child_0_status,
        "child1_status": child_1_status,
        "strong_scores": scores,
        "expert_choice": int(expert_choice),
    }
    return metadata


def _branch_child(node: BBNode, branch_var: int, branch_value: int, next_node_id: int, tolerance: float) -> BBNode | None:
    child_lb, child_ub, pre_status = make_strong_branch_child_bounds(
        node.lb,
        node.ub,
        branch_var,
        branch_value,
        tolerance,
    )
    if pre_status == "infeasible":
        return None
    return BBNode(
        node_id=next_node_id,
        depth=node.depth + 1,
        lb=child_lb,
        ub=child_ub,
        parent_id=node.node_id,
        branch_var=branch_var,
        branch_var_group="y",
        branch_value=float(branch_value),
        branch_direction=f"={branch_value}",
    )


def collect_instance_samples(
    generated: FamilyInstance,
    out_dir: str | Path,
    dataset_id: str = "fixed_charge_branching",
    lp_backend: str = "scipy_highs",
    scoring_mode: str = "product",
    epsilon: float = 1e-6,
    mu: float = 0.25,
    max_nodes_per_instance: int = 80,
    max_decisions_per_instance: int = 20,
    time_limit_per_instance: float = 30.0,
    max_lp_candidates: int | None = None,
    use_matrix_presolve: bool = True,
    tolerance: float = 1e-8,
    git_commit: str | None = None,
    max_nodes: int | None = None,
    max_decisions: int | None = None,
    control_policy=None,
    control_policy_name: str = "strong_branching",
    round_id: int = 0,
) -> InstanceCollectionResult:
    if max_nodes is not None:
        max_nodes_per_instance = int(max_nodes)
    if max_decisions is not None:
        max_decisions_per_instance = int(max_decisions)
    start = perf_counter()
    problem = generated.problem
    split = generated.parameters["split"]
    sample_dir = Path(out_dir) / split
    sample_dir.mkdir(parents=True, exist_ok=True)
    lp_solver = get_lp_relaxation_solver(lp_backend)
    policy = StrongBranchingPolicy(
        lp_backend=lp_backend,
        max_lp_candidates=max_lp_candidates,
        use_matrix_presolve=use_matrix_presolve,
        score_mode=scoring_mode,
        epsilon=epsilon,
        mu=mu,
    )
    solver_config = {
        "max_nodes_per_instance": int(max_nodes_per_instance),
        "max_decisions_per_instance": int(max_decisions_per_instance),
        "time_limit_per_instance": float(time_limit_per_instance),
        "max_lp_candidates": max_lp_candidates,
        "use_matrix_presolve": bool(use_matrix_presolve),
    }

    pool = BestBoundNodePool()
    next_node_id = 1
    formal_lp_count = 0
    formal_nodes_processed = 0
    incumbent_value: float | None = None
    samples_written = 0
    sample_paths: list[str] = []
    sample_summaries: list[dict] = []
    status = "optimal"
    limit_reason = ""

    root = BBNode(node_id=0, depth=0, lb=problem.lb.copy(), ub=problem.ub.copy(), branch_direction="root")
    root_lp = lp_solver(problem, root.lb, root.ub, tolerance, max_lp_candidates, use_matrix_presolve)
    formal_lp_count += 1
    formal_nodes_processed += 1
    if root_lp.status != "optimal":
        return InstanceCollectionResult(
            instance_id=generated.instance_id,
            split=split,
            family_name=generated.parameters.get("family_name", getattr(generated, "family_name", "unknown")),
            scale_group=generated.parameters.get("scale_group", getattr(generated, "scale_group", "unknown")),
            seed=generated.seed,
            units=generated.units,
            status=root_lp.status,
            samples_written=0,
            formal_lp_count=formal_lp_count,
            formal_nodes_processed=formal_nodes_processed,
            probe_lp_count=0,
            probe_runtime_sec=0.0,
            runtime_sec=perf_counter() - start,
            skipped=True,
            skip_reason=f"root_lp_{root_lp.status}",
            limit_reason=root_lp.status if root_lp.status != "infeasible" else "",
            sample_paths=[],
            sample_summaries=[],
        )
    pool.push(root, root_lp, float(root_lp.objective_value))

    while len(pool) > 0:
        elapsed = perf_counter() - start
        if elapsed >= time_limit_per_instance:
            status = "LIMIT"
            limit_reason = "time_limit_per_instance"
            break
        if formal_lp_count >= max_nodes_per_instance:
            status = "LIMIT"
            limit_reason = "max_nodes_per_instance"
            break
        if samples_written >= max_decisions_per_instance:
            status = "LIMIT"
            limit_reason = "max_decisions_per_instance"
            break

        node, lp, node_bound = pool.pop()
        if incumbent_value is not None and float(node_bound) <= incumbent_value + tolerance:
            continue
        if lp.status != "optimal" or lp.x is None:
            continue
        if is_binary_solution(lp.x, problem.binary_indices, tolerance):
            candidate = lp.x.copy()
            for index in problem.binary_indices:
                candidate[index] = round(candidate[index])
            if check_feasibility(problem, candidate, node.lb, node.ub, tolerance):
                value = float(problem.internal_c @ candidate)
                if incumbent_value is None or value > incumbent_value + tolerance:
                    incumbent_value = value
            continue

        candidate_indices = fractional_binary_candidates(lp.x, problem.binary_indices, tolerance)
        if not candidate_indices:
            continue
        context = BranchingContext(
            problem=problem,
            node_id=node.node_id,
            node_depth=node.depth,
            node_lb=node.lb.copy(),
            node_ub=node.ub.copy(),
            lp_result=lp,
            candidate_indices=candidate_indices,
            incumbent_internal_value=incumbent_value,
            current_node_internal_bound=float(node_bound),
            tolerance=tolerance,
        )
        expert_choice = policy.select_variable(context)
        records = list(policy.last_records)
        if not records:
            continue
        if control_policy is None:
            branch_choice = int(expert_choice)
        else:
            branch_choice = int(control_policy.select_variable(context))
            if branch_choice not in candidate_indices:
                raise ValueError("control policy returned an index outside candidate_indices")

        prefix = "" if int(round_id) == 0 else f"round_{int(round_id)}_"
        sample_path = sample_dir / f"{prefix}{generated.instance_id}_node_{node.node_id}.npz"
        save_sample(
            sample_path,
            _sample_arrays(problem, node, lp, records),
            _sample_metadata(
                dataset_id=dataset_id,
                split=split,
                generated=generated,
                node=node,
                lp=lp,
                node_bound=float(node_bound),
                incumbent_value=incumbent_value,
                records=records,
                expert_choice=expert_choice,
                scoring_mode=scoring_mode,
                epsilon=epsilon,
                mu=mu,
                tolerance=tolerance,
                lp_backend=lp_backend,
                solver_config=solver_config,
                git_commit=git_commit,
                probe_lp_count=sum(r.child_0_lp_solved + r.child_1_lp_solved for r in records),
                probe_runtime_sec=sum(r.child_0_time_sec + r.child_1_time_sec for r in records),
                round_id=round_id,
                control_policy_name=control_policy_name,
                control_branching_variable=branch_choice,
            ),
        )
        samples_written += 1
        sample_paths.append(str(sample_path))
        summary = {
            "path": str(sample_path),
            "node_id": int(node.node_id),
            "node_depth": int(node.depth),
            "candidate_indices": [int(r.candidate_index) for r in records],
            "candidate_lp_values": [float(r.candidate_lp_value) for r in records],
            "child_0_status": [r.child_0_status for r in records],
            "child_0_bound": [None if r.child_0_bound is None else float(r.child_0_bound) for r in records],
            "child_1_status": [r.child_1_status for r in records],
            "child_1_bound": [None if r.child_1_bound is None else float(r.child_1_bound) for r in records],
            "delta_0": [float(r.delta_0) for r in records],
            "delta_1": [float(r.delta_1) for r in records],
            "expert_score": [float(r.score) for r in records],
            "expert_selected_variable": int(expert_choice),
            "control_branching_variable": int(branch_choice),
            "round_id": int(round_id),
        }
        sample_summaries.append(summary)

        for branch_value in (1, 0):
            child = _branch_child(node, int(branch_choice), branch_value, next_node_id, tolerance)
            if child is None:
                continue
            next_node_id += 1
            child_lp = _solve_child_lp(lp_solver, problem, child, tolerance, max_lp_candidates, use_matrix_presolve)
            formal_lp_count += 1
            formal_nodes_processed += 1
            if child_lp.status in {"candidate_limit", "node_limit", "time_limit"}:
                status = "LIMIT"
                limit_reason = child_lp.status
                break
            if child_lp.status != "optimal" or child_lp.x is None:
                continue
            child_bound = float(child_lp.objective_value)
            if incumbent_value is not None and child_bound <= incumbent_value + tolerance:
                continue
            if is_binary_solution(child_lp.x, problem.binary_indices, tolerance):
                candidate = child_lp.x.copy()
                for index in problem.binary_indices:
                    candidate[index] = round(candidate[index])
                if check_feasibility(problem, candidate, child.lb, child.ub, tolerance):
                    value = float(problem.internal_c @ candidate)
                    if incumbent_value is None or value > incumbent_value + tolerance:
                        incumbent_value = value
                continue
            pool.push(child, child_lp, child_bound)
        if status == "LIMIT":
            break

    skipped = samples_written == 0
    skip_reason = "no_branch_decisions" if skipped and not limit_reason else ""
    return InstanceCollectionResult(
        instance_id=generated.instance_id,
        split=split,
        family_name=generated.parameters.get("family_name", getattr(generated, "family_name", "unknown")),
        scale_group=generated.parameters.get("scale_group", getattr(generated, "scale_group", "unknown")),
        seed=generated.seed,
        units=generated.units,
        status=status,
        samples_written=samples_written,
        formal_lp_count=formal_lp_count,
        formal_nodes_processed=formal_nodes_processed,
        probe_lp_count=int(policy.probe_lp_solved),
        probe_runtime_sec=float(policy.probe_time_sec),
        runtime_sec=perf_counter() - start,
        skipped=skipped,
        skip_reason=skip_reason,
        limit_reason=limit_reason,
        sample_paths=sample_paths,
        sample_summaries=sample_summaries,
    )


def _load_config(path: str | Path) -> dict:
    config = dict(DEFAULT_CONFIG)
    config_path = Path(path)
    loaded = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(loaded)
    return config


def _prepare_output_dir(out_dir: Path, overwrite: bool, resume: bool) -> dict:
    manifest_path = out_dir / "manifest.json"
    if overwrite and out_dir.exists():
        for child in out_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        if not resume and not overwrite:
            raise ValueError(f"{manifest_path} already exists; use --resume or --overwrite")
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {"instances": [], "samples": [], "skipped": []}


def _split_family_counts(manifest: dict) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for row in manifest.get("instances", []):
        if row.get("skipped"):
            continue
        key = (row["split"], row.get("family_name", "unknown"))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _expanded_split_specs(config: dict) -> list[dict]:
    default_families = list(config.get("families") or sorted(FAMILY_REGISTRY))
    specs = []
    for split, raw_spec in dict(config["splits"]).items():
        families = list(raw_spec.get("families") or default_families)
        sizes = list(raw_spec.get("sizes") or [4, 5])
        scale_group = str(raw_spec.get("scale_group", split))
        count = int(raw_spec.get("instances_per_family", 1))
        for family_name in families:
            specs.append(
                {
                    "split": split,
                    "family_name": family_name,
                    "sizes": sizes,
                    "scale_group": scale_group,
                    "target": count,
                }
            )
    return specs


def collect_dataset(
    config: dict,
    resume: bool = False,
    overwrite: bool = False,
    max_instances: int | None = None,
    verbose: bool = False,
    report_dir: str | Path | None = None,
) -> dict:
    out_dir = Path(config["output_dir"])
    manifest = _prepare_output_dir(out_dir, overwrite=overwrite, resume=resume)
    dataset_id = str(config["dataset_id"])
    git_commit = _git_commit()
    manifest.update(
        {
            "dataset_id": dataset_id,
            "output_dir": str(out_dir),
            "config": config,
            "git_commit": git_commit,
            "updated_at": _now_iso(),
        }
    )
    existing_ids = {row["instance_id"] for row in manifest.get("instances", [])}
    accepted_counts = _split_family_counts(manifest)
    split_specs = _expanded_split_specs(config)
    max_attempts = int(config.get("max_generation_attempts", 100))
    master_seed = int(config["master_seed"])
    attempted = 0
    accepted_total_start = sum(accepted_counts.values())

    for spec_index, spec in enumerate(split_specs):
        split = spec["split"]
        family_name = spec["family_name"]
        family = get_family(family_name)
        target_count = int(spec["target"])
        key = (split, family_name)
        local_attempt = 0
        while accepted_counts.get(key, 0) < target_count and local_attempt < max_attempts:
            if max_instances is not None and sum(accepted_counts.values()) - accepted_total_start >= max_instances:
                break
            size = int(spec["sizes"][local_attempt % len(spec["sizes"])])
            seed = master_seed + spec_index * 1_000_000 + local_attempt
            local_attempt += 1
            attempted += 1
            generated = family.generate(seed=seed, size=size, split=split, scale_group=spec["scale_group"])
            if generated.instance_id in existing_ids:
                continue
            result = collect_instance_samples(
                generated,
                out_dir=out_dir,
                dataset_id=dataset_id,
                lp_backend=config["lp_backend"],
                scoring_mode=config["scoring_mode"],
                epsilon=float(config["epsilon"]),
                mu=float(config["mu"]),
                max_nodes_per_instance=int(config["max_nodes_per_instance"]),
                max_decisions_per_instance=int(config["max_decisions_per_instance"]),
                time_limit_per_instance=float(config["time_limit_per_instance"]),
                max_lp_candidates=config.get("max_lp_candidates"),
                use_matrix_presolve=bool(config["use_matrix_presolve"]),
                tolerance=float(config["tolerance"]),
                git_commit=git_commit,
            )
            row = result.to_manifest_row()
            if result.skipped:
                manifest.setdefault("skipped", []).append(row)
                if verbose:
                    print(f"skipped {generated.instance_id}: {result.skip_reason or result.status}")
                continue
            manifest.setdefault("instances", []).append(row)
            for sample_path in result.sample_paths or []:
                manifest.setdefault("samples", []).append(
                    {
                        "path": sample_path,
                        "split": split,
                        "family_name": result.family_name,
                        "scale_group": result.scale_group,
                        "instance_id": result.instance_id,
                        "seed": result.seed,
                        "units": result.units,
                    }
                )
            existing_ids.add(generated.instance_id)
            accepted_counts[key] = accepted_counts.get(key, 0) + 1
            if verbose:
                print(f"accepted {generated.instance_id}: samples={result.samples_written} status={result.status}")

    manifest["updated_at"] = _now_iso()
    manifest["accepted_counts"] = {f"{split}:{family}": count for (split, family), count in sorted(accepted_counts.items())}
    manifest["target_counts"] = {
        f"{spec['split']}:{spec['family_name']}": int(spec["target"])
        for spec in split_specs
    }
    manifest["attempted_instances_this_run"] = attempted
    manifest["total_samples"] = len(manifest.get("samples", []))
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    _write_jsonl(out_dir / "manifest.jsonl", manifest.get("samples", []))
    if report_dir is not None:
        write_dataset_report(out_dir, report_dir)
    return manifest


def _format_float(value: float) -> str:
    return f"{float(value):.6g}"


def dataset_statistics(out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    stats = {
        "dataset_id": manifest.get("dataset_id", ""),
        "output_dir": str(out_dir),
        "split_instance_counts": {"train": 0, "validation": 0, "test": 0},
        "split_sample_counts": {"train": 0, "validation": 0, "test": 0},
        "candidate_count_min": 0,
        "candidate_count_max": 0,
        "candidate_count_mean": 0.0,
        "units_distribution": {},
        "infeasible_probe_ratio": 0.0,
        "average_probe_lp_per_instance": 0.0,
        "average_collection_time_per_instance": 0.0,
        "skipped_count": len(manifest.get("skipped", [])),
        "skipped_reasons": {},
        "limit_instance_count": 0,
        "probe_lp_total": 0,
        "probe_runtime_total_sec": 0.0,
        "sample_count": 0,
    }
    for row in manifest.get("instances", []):
        split = row["split"]
        stats["split_instance_counts"][split] = stats["split_instance_counts"].get(split, 0) + 1
        stats["units_distribution"][str(row["units"])] = stats["units_distribution"].get(str(row["units"]), 0) + 1
        stats["probe_lp_total"] += int(row["probe_lp_count"])
        stats["probe_runtime_total_sec"] += float(row["probe_runtime_sec"])
        if row.get("status") == "LIMIT":
            stats["limit_instance_count"] += 1
    for row in manifest.get("skipped", []):
        reason = row.get("skip_reason") or row.get("status") or "unknown"
        stats["skipped_reasons"][reason] = stats["skipped_reasons"].get(reason, 0) + 1

    if manifest.get("instances"):
        stats["average_probe_lp_per_instance"] = stats["probe_lp_total"] / len(manifest["instances"])
        stats["average_collection_time_per_instance"] = sum(float(r["runtime_sec"]) for r in manifest["instances"]) / len(manifest["instances"])

    try:
        dataset = BranchingDataset.from_dir(out_dir)
        dataset.assert_disjoint_splits()
    except ValueError:
        return stats

    candidate_counts = []
    infeasible_probe_count = 0
    total_probe_children = 0
    for sample in dataset:
        split = sample["metadata"]["split"]
        stats["split_sample_counts"][split] = stats["split_sample_counts"].get(split, 0) + 1
        n_candidates = len(sample["arrays"]["candidate_indices"])
        candidate_counts.append(n_candidates)
        statuses = list(sample["metadata"]["child_0_status"]) + list(sample["metadata"]["child_1_status"])
        infeasible_probe_count += sum(1 for status in statuses if status == "infeasible")
        total_probe_children += len(statuses)
    stats["sample_count"] = len(candidate_counts)
    if candidate_counts:
        stats["candidate_count_min"] = int(min(candidate_counts))
        stats["candidate_count_max"] = int(max(candidate_counts))
        stats["candidate_count_mean"] = float(np.mean(candidate_counts))
    if total_probe_children:
        stats["infeasible_probe_ratio"] = infeasible_probe_count / total_probe_children
    return stats


def write_dataset_report(out_dir: str | Path, report_dir: str | Path) -> dict:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    stats = dataset_statistics(out_dir)
    csv_path = report_dir / "dataset_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in stats.items():
            writer.writerow([key, json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, dict) else value])

    md_path = report_dir / "dataset_report.md"
    lines = [
        "# Learning Branching Dataset Report",
        "",
        f"- dataset_id: `{stats['dataset_id']}`",
        f"- output_dir: `{stats['output_dir']}`",
        f"- total decision samples: `{stats['sample_count']}`",
        f"- split instance counts: `{stats['split_instance_counts']}`",
        f"- split sample counts: `{stats['split_sample_counts']}`",
        f"- candidate count min / mean / max: `{stats['candidate_count_min']} / {_format_float(stats['candidate_count_mean'])} / {stats['candidate_count_max']}`",
        f"- units distribution: `{stats['units_distribution']}`",
        f"- infeasible probe ratio: `{_format_float(stats['infeasible_probe_ratio'])}`",
        f"- probe LP total: `{stats['probe_lp_total']}`",
        f"- probe runtime total sec: `{_format_float(stats['probe_runtime_total_sec'])}`",
        f"- average probe LP per instance: `{_format_float(stats['average_probe_lp_per_instance'])}`",
        f"- average collection time per instance sec: `{_format_float(stats['average_collection_time_per_instance'])}`",
        f"- skipped instances: `{stats['skipped_count']}`",
        f"- skipped reasons: `{stats['skipped_reasons']}`",
        f"- LIMIT instances: `{stats['limit_instance_count']}`",
        "",
        "说明：probe LP 是 strong branching 为候选变量额外求解的试探 LP，不计入正式 B&B 节点 LP。",
        "",
        "当前限制：如果 `candidate_count_max=1` 或多候选比例过低，说明该批数据只能验证采集链路，不能作为有效分支策略训练数据。",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return stats


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Collect strong-branching expert data for multi-family binary MILPs.")
    parser.add_argument("--config", default="ml_branching/configs/smoke.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    config = _load_config(args.config)
    manifest = collect_dataset(
        config,
        resume=args.resume,
        overwrite=args.overwrite,
        max_instances=args.max_instances,
        verbose=args.verbose,
        report_dir=None,
    )
    summary = {
        "dataset_id": manifest["dataset_id"],
        "output_dir": manifest["output_dir"],
        "accepted_counts": manifest["accepted_counts"],
        "total_samples": manifest["total_samples"],
        "skipped_count": len(manifest.get("skipped", [])),
    }
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return manifest


if __name__ == "__main__":
    main()
