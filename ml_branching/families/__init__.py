from __future__ import annotations

from .activated_resource_allocation import ActivatedResourceAllocationFamily
from .base import FamilyInstance, MILPFamily, instance_stats
from .capacity_expansion import CapacityExpansionFamily
from .facility_location import FacilityLocationFamily
from .fixed_charge_multi_resource import FixedChargeMultiResourceFamily
from .random_sparse_block import RandomSparseBlockFamily
from .unit_commitment import UnitCommitmentFamily


FAMILY_REGISTRY = {
    cls.family_name: cls()
    for cls in (
        FixedChargeMultiResourceFamily,
        UnitCommitmentFamily,
        CapacityExpansionFamily,
        FacilityLocationFamily,
        ActivatedResourceAllocationFamily,
        RandomSparseBlockFamily,
    )
}


def get_family(name: str) -> MILPFamily:
    try:
        return FAMILY_REGISTRY[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown family_name={name!r}; available={sorted(FAMILY_REGISTRY)}") from exc


def reconstruct_instance(parameters: dict) -> FamilyInstance:
    family_name = str(parameters["family_name"])
    family = get_family(family_name)
    return family.generate(
        seed=int(parameters["seed"]),
        size=int(parameters.get("size", parameters.get("units"))),
        split=str(parameters.get("split", "reconstructed")),
        scale_group=str(parameters.get("scale_group", "reconstructed")),
    )


__all__ = [
    "ActivatedResourceAllocationFamily",
    "CapacityExpansionFamily",
    "FacilityLocationFamily",
    "FamilyInstance",
    "FAMILY_REGISTRY",
    "FixedChargeMultiResourceFamily",
    "MILPFamily",
    "RandomSparseBlockFamily",
    "UnitCommitmentFamily",
    "get_family",
    "instance_stats",
    "reconstruct_instance",
]
