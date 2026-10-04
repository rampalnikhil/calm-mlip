"""Repeated scientific interfaces keep one authoritative atoms artifact."""

from dataclasses import replace

import pytest

from current_structure_fixtures import current_atoms_payload
from calm.project.application.artifacts import ArtifactsService
from calm.project.application.derived_interfaces import DerivedInterfaceService
from calm.project.application.runs import RunsService
from calm.project.infrastructure.artifacts.fs_store import FSArtifactStore


def test_interface_atoms_reuse_after_another_search_and_reopen(
    tmp_path, sqlite_uow_factory, prototype_graph_factory, prototype_payload_factory,
):
    prototype_graph_factory(
        prototype_payload=prototype_payload_factory(include_supercells=True),
    )
    with sqlite_uow_factory() as uow:
        summary, = uow.prototypes.query()
        prototype = uow.prototypes.get_by_uid_full(summary.uid_full)
    store = FSArtifactStore(tmp_path / "artifacts")

    def service():
        return DerivedInterfaceService(
            uow_factory=sqlite_uow_factory,
            artifacts=ArtifactsService(uow_factory=sqlite_uow_factory, store=store),
        )

    atoms = current_atoms_payload()
    first = service().create(prototype=prototype.uid_full, atoms=atoms)
    repeated = service().create(prototype=prototype.uid_full, atoms=atoms)
    assert repeated.spec == first.spec
    files = {p: p.read_bytes() for p in (tmp_path / "artifacts").rglob("*.json")}

    second_run = RunsService(sqlite_uow_factory()).create(
        run_type="prototype", spec={"search": "second"},
    )
    with sqlite_uow_factory() as uow:
        uow.prototypes.upsert_many([
            replace(prototype, run_uid_full=second_run.uid_full),
        ])

    repeated = service().create(prototype=prototype.uid_full, atoms=atoms)
    assert repeated.uid_full == first.uid_full
    assert repeated.spec == first.spec
    assert {p: p.read_bytes() for p in files} == files
    assert set((tmp_path / "artifacts").rglob("*.json")) == set(files)
    with sqlite_uow_factory() as uow:
        assert uow.derived_interfaces.get_by_uid_full(first.uid_full).spec == first.spec
        assert len(uow.artifacts.list_for_run(prototype.run_uid_full)) == 1
        assert uow.artifacts.list_for_run(second_run.uid_full) == []
        edges = uow.edges.list(src_uid_full=second_run.uid_full, kind="run_to_interface")
        assert any(edge.dst_uid_full == first.uid_full for edge in edges)

    # Explicit producing runs and interface seeds must also reuse the artifact.
    explicit = service().create(
        prototype=first.uid_full, atoms=atoms, artifact_run=second_run.uid_full,
    )
    assert explicit.spec == first.spec

    changed = current_atoms_payload(scaled_positions=((0.125, 0.0, 0.0),))
    with pytest.raises(ValueError, match="conflicting atoms artifacts"):
        service().create(prototype=prototype.uid_full, atoms=changed)
    with sqlite_uow_factory() as uow:
        assert uow.derived_interfaces.get_by_uid_full(first.uid_full).spec == first.spec
        assert uow.artifacts.list_for_run(second_run.uid_full) == []
    assert set((tmp_path / "artifacts").rglob("*.json")) == set(files)


def test_spec_only_interface_can_receive_its_first_atoms(
    tmp_path, sqlite_uow_factory, prototype_graph_factory, prototype_payload_factory,
):
    prototype_graph_factory(
        prototype_payload=prototype_payload_factory(include_supercells=True),
    )
    with sqlite_uow_factory() as uow:
        summary, = uow.prototypes.query()
        prototype = uow.prototypes.get_by_uid_full(summary.uid_full)
    service = DerivedInterfaceService(
        uow_factory=sqlite_uow_factory,
        artifacts=ArtifactsService(
            uow_factory=sqlite_uow_factory, store=FSArtifactStore(tmp_path / "artifacts"),
        ),
    )
    spec_only = service.create(prototype=prototype.uid_full)
    assert spec_only.atoms_artifact_uid is None
    populated = service.create(prototype=spec_only.uid_full, atoms=current_atoms_payload())
    assert populated.uid_full == spec_only.uid_full
    assert populated.atoms_artifact_uid is not None
