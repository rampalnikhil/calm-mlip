"""Failed attempts remain retryable; completed results remain immutable."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from calm.project.application.derived_interfaces import DerivedInterfaceService
from calm.project.application.followups import relaxation as relaxation_module
from calm.project.application.followups import reference_energy as reference_module
from calm.project.application.followups import thermodynamics as thermo_module
from calm.project.application.followups.energy import EnergyOrchestrator
from calm.project.application.followups.energy_backends import EnergyComputeResult
from calm.project.application.followups.relaxation import RelaxationOrchestrator
from calm.project.application.followups.relaxation_backends import RelaxationComputeResult
from calm.project.application.followups.reference_energy import ReferenceEnergyOrchestrator
from calm.project.application.followups.thermodynamics import ThermodynamicOrchestrator
from energy_result_fixtures import reference_energy_payload


class _Atoms:
    def get_cell(self):
        return np.diag([3.0, 3.0, 10.0])


class _Backend:
    name = "retry-integrity"

    def __init__(self, *, relaxation=False):
        self.relaxation = relaxation
        self.broken = set()
        self.calls = []
        self.message = "first failure"

    def compute(self, *, target_uid, **kwargs):
        self.calls.append(target_uid)
        if target_uid in self.broken:
            raise RuntimeError(self.message)
        if self.relaxation:
            return RelaxationComputeResult(
                final_energy=-1.0, n_steps=1, relaxed_params={}, summary={},
                artifact_payloads=[], converged=True, max_force=0.0,
                max_optimizer_residual=0.0,
            )
        return EnergyComputeResult(energy=-1.0, n_steps=1, summary={}, artifact_payloads=[])


def _targets(factory, graph, payload):
    graph(prototype_payload=payload(include_supercells=True))
    with factory() as uow:
        prototype, = uow.prototypes.query()
    service = DerivedInterfaceService(uow_factory=factory)
    return [service.create(prototype=prototype.uid_full, vacuum=v).uid_full
            for v in (10.0, 15.0)]


def _stage(kind, factory, targets, monkeypatch):
    backend = _Backend(relaxation=kind == "relaxation")
    if kind == "relaxation":
        return (RelaxationOrchestrator(factory()).with_backend(backend),
                {"prototypes": targets}, backend, relaxation_module, "_existing_followups")
    if kind == "reference":
        # Isolate structure preparation from persistence, without requiring ASE.
        def prepare(uow, *, interface_uid_full, **kwargs):
            interface = uow.derived_interfaces.get_by_uid_full(interface_uid_full)
            row = reference_energy_payload(
                reference_uid_full=interface_uid_full,
                reference_kind="strained_bulk_a", formula_id="interface_excess_strained_bulk",
                energy_eV=-1.0, energy_eV_per_formula_unit=-1.0,
                reference_formula_units=1, interface_formula_units=1,
            )
            return [reference_module._PreparedReferenceTarget(
                source_interface_uid_full=interface_uid_full,
                prototype_uid_full=interface.prototype_uid_full,
                reference_uid_full=interface_uid_full, reference_kind="strained_bulk_a",
                side="a", atoms=_Atoms(), metadata=row["reference"]["metadata"],
            )]
        monkeypatch.setattr(reference_module, "_prepare_interface_targets", prepare)
        return (ReferenceEnergyOrchestrator(factory()).with_backend(backend),
                {"interfaces": targets, "formula": "interface_excess_strained_bulk"},
                backend, reference_module, "_existing")
    raw_run, raw = EnergyOrchestrator(
        factory(), target_atoms_loader=lambda uid: _Atoms(),
    ).with_backend(backend).run_stage(prototypes=targets)
    assert raw_run.status == "done"
    service = ThermodynamicOrchestrator(factory())
    original = service._references_for_target

    def references(values, uid):
        if uid in backend.broken:
            raise RuntimeError(backend.message)
        return original(values, uid)

    monkeypatch.setattr(service, "_references_for_target", references)
    return (service, {
        "raw_energy_results": [row.followup_uid for row in raw],
        "convention": {"formula": "interface_excess_strained_bulk", "n_interfaces": 1},
        "references": {"bulk_a_eV_per_formula_unit": -1.0,
                       "bulk_b_eV_per_formula_unit": -1.0,
                       "n_formula_units_a": 1, "n_formula_units_b": 1},
    }, backend, thermo_module, "_existing")


@pytest.mark.filterwarnings("ignore:Relaxation backend error:UserWarning")
@pytest.mark.parametrize("kind", ["relaxation", "reference", "thermodynamic"])
def test_failed_attempts_retry_and_completed_results_cannot_change(
    sqlite_uow_factory, prototype_graph_factory, prototype_payload_factory, monkeypatch, kind,
):
    targets = _targets(sqlite_uow_factory, prototype_graph_factory, prototype_payload_factory)
    service, inputs, backend, _, _ = _stage(kind, sqlite_uow_factory, targets, monkeypatch)
    backend.broken.add(targets[0])
    run, results = service.run_stage(**inputs)
    assert run.status == "failed"
    assert sorted(r.status for r in results) == ["completed", "failed"]
    result_uids = {r.followup_uid for r in results}
    with sqlite_uow_factory() as uow:
        initial_rows = uow.followups.list(run_uid_full=run.uid_full)
        initial_ids = {r.uid_full: r.id_short for r in initial_rows}
        failed_row, = [row for row in initial_rows if row.status == "failed"]
        incompatible = deepcopy(failed_row.payload)
        if kind == "relaxation":
            incompatible["backend_identity"]["revision"] = "different"
        elif kind == "reference":
            incompatible["backend"]["identity"]["revision"] = "different"
        else:
            incompatible["convention"]["n_interfaces"] = 2
        with pytest.raises(ValueError, match="different scientific results"):
            uow.followups.upsert_many([replace(failed_row, payload=incompatible)])
    backend.message = "second failure"
    failed, _ = service.run_stage(**inputs, partial_resume=True)
    assert failed.uid_full == run.uid_full and failed.status == "failed"
    with sqlite_uow_factory() as uow:
        row, = [r for r in uow.followups.list(run_uid_full=run.uid_full) if r.status == "failed"]
        assert row.payload["failure"]["message"] == "second failure"

    # Also recover mixed runs written as done by older CALM versions.
    service._runs.mark_done(run.uid_full, progress={"n_failed": 1})
    backend.broken.clear()
    retried, results = service.run_stage(**inputs, partial_resume=True)
    assert retried.uid_full == run.uid_full and retried.status == "done"
    assert sorted(r.status for r in results) == ["completed", "skipped"]
    assert {r.followup_uid for r in results} == result_uids
    with sqlite_uow_factory() as uow:
        rows = uow.followups.list(run_uid_full=run.uid_full)
        assert {r.status for r in rows} == {"done"}
        assert {r.uid_full: r.id_short for r in rows} == initial_ids
        changed = deepcopy(rows[0].payload)
        # An additional workflow annotation must not mutate completed results.
        if kind == "thermodynamic":
            changed["references"]["metadata"]["changed"] = True
        else:
            changed["summary"]["changed"] = True
        with pytest.raises(ValueError, match="different scientific results"):
            uow.followups.upsert_many([replace(rows[0], payload=changed)])


@pytest.mark.parametrize("kind", ["relaxation", "reference", "thermodynamic"])
@pytest.mark.parametrize("point", ["resume", "persistence"])
def test_followup_infrastructure_errors_leave_failed_runs(
    sqlite_uow_factory, prototype_graph_factory, prototype_payload_factory,
    monkeypatch, kind, point,
):
    targets = _targets(sqlite_uow_factory, prototype_graph_factory, prototype_payload_factory)
    service, inputs, _, module, lookup = _stage(kind, sqlite_uow_factory, targets, monkeypatch)

    def fail(*args, **kwargs):
        raise OSError("infrastructure unavailable")

    monkeypatch.setattr(service if point == "resume" else module,
                        lookup if point == "resume" else "persist_followups_with_edges", fail)
    with pytest.raises(OSError, match="infrastructure unavailable"):
        service.run_stage(**inputs, partial_resume=True)
    with sqlite_uow_factory() as uow:
        run_type = {"relaxation": "relaxation_stage", "reference": "reference_energy",
                    "thermodynamic": "thermodynamic_derivation"}[kind]
        run, = uow.runs.list(run_type=run_type)
        assert run.status == "failed"
        assert run.error["exception_type"] == "OSError"
        assert uow.followups.list(run_uid_full=run.uid_full) == []


def test_relaxation_warning_as_error_finalizes_run(
    sqlite_uow_factory, prototype_graph_factory, prototype_payload_factory,
):
    targets = _targets(sqlite_uow_factory, prototype_graph_factory, prototype_payload_factory)
    backend = _Backend(relaxation=True)
    backend.broken.update(targets)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(UserWarning, match="Relaxation backend error"):
            RelaxationOrchestrator(sqlite_uow_factory()).with_backend(backend).run_stage(
                prototypes=targets,
            )
    with sqlite_uow_factory() as uow:
        run, = uow.runs.list(run_type="relaxation_stage")
        assert run.status == "failed"
        assert run.error["exception_type"] == "UserWarning"


def test_repeated_raw_energy_failure_updates_diagnostics(
    sqlite_uow_factory, prototype_graph_factory, prototype_payload_factory,
):
    targets = _targets(sqlite_uow_factory, prototype_graph_factory, prototype_payload_factory)
    message = ["first failure"]

    def load(uid):
        raise FileNotFoundError(message[0])

    service = EnergyOrchestrator(sqlite_uow_factory(), target_atoms_loader=load).with_backend(_Backend())
    first, _ = service.run_stage(prototypes=targets[:1])
    message[0] = "second failure"
    second, _ = service.run_stage(prototypes=targets[:1], partial_resume=True)
    assert first.uid_full == second.uid_full and second.status == "failed"
    with sqlite_uow_factory() as uow:
        row, = uow.followups.list(run_uid_full=second.uid_full)
        assert row.payload["failure"]["message"] == "second failure"
