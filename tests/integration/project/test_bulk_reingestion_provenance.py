"""Scientific identity reuse must not erase optimization provenance."""

from copy import deepcopy

import pytest

from bulk_record_fixtures import current_bulk_payload
from calm.calculators.spec import CalculatorSpec
from calm.project.application.bulk_fingerprinting import BulkFingerprintService
from calm.project.application.bulks import BulksService


@pytest.mark.parametrize("structure_backed", [False, True])
def test_bulk_reingestion_preserves_provenance_and_rejects_conflicts(
    sqlite_uow_factory, monkeypatch, structure_backed,
):
    # Supply a valid current fingerprint without requiring a symmetry backend;
    # persistence, identity, calculator registration, and reopening are real.
    payload = current_bulk_payload()
    monkeypatch.setattr(
        BulkFingerprintService, "assemble_structure_payload",
        lambda self, structure, **kwargs: deepcopy(payload),
    )
    inputs = {"structure": object()} if structure_backed else {"payload": {"material": "Al"}}
    service = BulksService(uow_factory=sqlite_uow_factory)
    calculator = CalculatorSpec(family="ase", model="EMT")
    optimized = service.add_bulk(label="optimized", optimized_with=calculator, **inputs)
    restored = service.add_bulk(label="reimported", **inputs)
    assert restored.uid_full == optimized.uid_full
    assert restored.kind == "optimized"
    assert restored.optimized_with_calculator_uid_full == optimized.optimized_with_calculator_uid_full
    assert restored.calculator == "ase:EMT"

    same = service.add_bulk(label="same calculator", optimized_with=calculator, **inputs)
    assert same.optimized_with_calculator_uid_full == optimized.optimized_with_calculator_uid_full
    with pytest.raises(ValueError, match="optimization calculator"):
        service.add_bulk(
            label="conflict", optimized_with=CalculatorSpec(family="ase", model="LennardJones"),
            **inputs,
        )
    with sqlite_uow_factory() as uow:
        reopened = uow.bulks.get_by_uid_full(optimized.uid_full)
        assert reopened.kind == "optimized"
        assert reopened.label == "same calculator"
        assert reopened.optimized_with_calculator_uid_full == optimized.optimized_with_calculator_uid_full
        if structure_backed:
            summary, = uow.bulks.list()
            assert reopened.natoms == summary.natoms == payload["derived"]["n_atoms"] == 4


def test_reference_bulk_can_acquire_optimization_provenance(sqlite_uow_factory):
    service = BulksService(uow_factory=sqlite_uow_factory)
    reference = service.add_bulk(label="reference", payload={"material": "A"})
    optimized = service.add_bulk(
        label="optimized", payload={"material": "A"}, optimized_with="ase:EMT",
    )
    assert optimized.uid_full == reference.uid_full
    assert optimized.kind == "optimized"
    assert optimized.optimized_with_calculator_uid_full is not None
