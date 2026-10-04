"""Loading and persistence failures must leave durable terminal run states."""

from dataclasses import replace

import numpy as np
import pytest

from calm.project.application.derived_interfaces import DerivedInterfaceService
from calm.project.application.followups import energy as energy_module
from calm.project.application.followups.energy import EnergyOrchestrator
from calm.project.application.followups.energy_backends import EnergyComputeResult


class _Atoms:
    def get_cell(self):
        return np.diag([3.0, 3.0, 10.0])


class _Backend:
    name = "failure-finalization-test"

    def __init__(self):
        self.calls = []

    def compute(self, *, target_uid, **kwargs):
        self.calls.append(target_uid)
        return EnergyComputeResult(energy=-1.0, n_steps=1, summary={}, artifact_payloads=[])


def _interfaces(sqlite_uow_factory, prototype_graph_factory, prototype_payload_factory):
    prototype_graph_factory(
        prototype_payload=prototype_payload_factory(include_supercells=True),
    )
    with sqlite_uow_factory() as uow:
        prototype, = uow.prototypes.query()
    service = DerivedInterfaceService(uow_factory=sqlite_uow_factory)
    return [
        service.create(prototype=prototype.uid_full, vacuum=vacuum).uid_full
        for vacuum in (10.0, 15.0)
    ]


@pytest.mark.parametrize("exception", [FileNotFoundError, ValueError])
def test_loading_failure_records_followup_and_failed_run(
    sqlite_uow_factory, prototype_graph_factory, prototype_payload_factory, exception,
):
    targets = _interfaces(
        sqlite_uow_factory, prototype_graph_factory, prototype_payload_factory,
    )

    def load(uid):
        raise exception("cannot load authoritative atoms")

    backend = _Backend()
    run, results = EnergyOrchestrator(
        sqlite_uow_factory(), target_atoms_loader=load,
    ).with_backend(backend).run_stage(prototypes=targets[:1])
    assert run.status == "failed"
    assert run.error == {"n_failed": 1}
    assert [r.status for r in results] == ["failed"]
    assert backend.calls == []
    with sqlite_uow_factory() as uow:
        assert uow.runs.get_by_uid_full(run.uid_full).status == "failed"
        followup, = uow.followups.list(run_uid_full=run.uid_full, kind="energy_stage")
        assert followup.status == "failed"
        assert followup.payload["failure"]["exception_type"] == exception.__name__
        assert followup.payload["failure"]["message"] == "cannot load authoritative atoms"


def test_mixed_failure_preserves_success_and_partial_resume_retries_failed_target(
    sqlite_uow_factory, prototype_graph_factory, prototype_payload_factory,
):
    targets = _interfaces(
        sqlite_uow_factory, prototype_graph_factory, prototype_payload_factory,
    )
    broken = {targets[0]}

    def load(uid):
        if uid in broken:
            raise FileNotFoundError("missing atoms")
        return _Atoms()

    backend = _Backend()
    orchestrator = EnergyOrchestrator(
        sqlite_uow_factory(), target_atoms_loader=load,
    ).with_backend(backend)
    run, results = orchestrator.run_stage(prototypes=targets)
    assert run.status == "failed"
    assert [r.status for r in results] == ["failed", "completed"]
    assert backend.calls == [targets[1]]
    first_followup_uids = {r.target_uid: r.followup_uid for r in results}

    broken.clear()
    retried, results = orchestrator.run_stage(prototypes=targets, partial_resume=True)
    assert retried.uid_full == run.uid_full
    assert retried.status == "done"
    assert retried.error == {}
    assert [r.status for r in results] == ["completed", "skipped"]
    assert backend.calls == [targets[1], targets[0]]
    assert {r.target_uid: r.followup_uid for r in results} == first_followup_uids
    with sqlite_uow_factory() as uow:
        rows = uow.followups.list(run_uid_full=run.uid_full, kind="energy_stage")
        assert len(rows) == 2
        assert {row.status for row in rows} == {"done"}
        retried_result = next(row for row in rows if row.target_uid_full == targets[0])
        changed = replace(
            retried_result,
            best_energy=-2.0,
            payload={**retried_result.payload, "energy_eV": -2.0},
        )
        with pytest.raises(ValueError, match="different scientific results"):
            uow.followups.upsert_many([changed])


@pytest.mark.parametrize("failure_point", ["resume", "artifacts", "followups"])
def test_infrastructure_failure_marks_run_failed_and_propagates(
    sqlite_uow_factory, prototype_graph_factory, prototype_payload_factory,
    monkeypatch, failure_point,
):
    targets = _interfaces(
        sqlite_uow_factory, prototype_graph_factory, prototype_payload_factory,
    )
    orchestrator = EnergyOrchestrator(
        sqlite_uow_factory(), target_atoms_loader=lambda uid: _Atoms(),
    ).with_backend(_Backend())

    def fail(*args, **kwargs):
        raise OSError("persistence unavailable")

    if failure_point == "resume":
        monkeypatch.setattr(orchestrator, "_existing_followups", fail)
    else:
        name = "persist_artifact_payloads" if failure_point == "artifacts" else "persist_followups_with_edges"
        monkeypatch.setattr(energy_module, name, fail)
    with pytest.raises(OSError, match="persistence unavailable"):
        orchestrator.run_stage(prototypes=targets[:1], partial_resume=True)
    with sqlite_uow_factory() as uow:
        run, = uow.runs.list(run_type="energy_stage")
        assert run.status == "failed"
        assert run.error["exception_type"] == "OSError"
        assert run.error["message"] == "persistence unavailable"
        assert uow.followups.list(run_uid_full=run.uid_full, kind="energy_stage") == []
