"""Identity-bearing reads must reject changed bytes, even when JSON is valid."""

import json

import pytest

from calm.project.application.artifacts import ArtifactsService
from calm.project.application.derived_interfaces import DerivedInterfaceService
from calm.project.application.runs import RunsService
from calm.project.infrastructure.artifacts.fs_store import FSArtifactStore
from calm.project.runtime.workspace import Workspace
from current_structure_fixtures import current_atoms_payload


def _workspace(tmp_path, factory):
    store = FSArtifactStore(tmp_path)
    return (Workspace(out_dir=tmp_path, artifact_store=store, uow_factory=factory),
            ArtifactsService(uow_factory=factory, store=store))


@pytest.mark.parametrize("mode", ["bytes", "text", "json", "auto"])
def test_artifact_reads_verify_bytes_before_decoding(tmp_path, sqlite_uow_factory, mode):
    ws, artifacts = _workspace(tmp_path, sqlite_uow_factory)
    run = RunsService(sqlite_uow_factory()).create(run_type="integrity", spec={})
    ref = artifacts.put_json(run.uid_full, obj={"x": 1}, filename="data.json")
    path = ws.artifact_path(ref)
    original = path.read_bytes()
    expected = {"bytes": original, "text": original.decode(), "json": {"x": 1},
                "auto": {"x": 1}}[mode]
    assert ws.open_artifact(ref, mode=mode) == expected
    path.write_text('{"x": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="persisted identity"):
        ws.open_artifact(ref, mode=mode)
    # A raw URI and path-only request deliberately carry no content guarantee.
    assert ws.open_artifact(ref.uri, mode="json") == {"x": 2}
    assert ws.open_artifact(ref, mode="path") == path
    path.unlink()
    with pytest.raises(FileNotFoundError):
        ws.open_artifact(ref, mode=mode)


def test_interface_atoms_reject_coordinate_tampering(
    tmp_path, sqlite_uow_factory, prototype_graph_factory, prototype_payload_factory,
):
    prototype_graph_factory(prototype_payload=prototype_payload_factory(include_supercells=True))
    with sqlite_uow_factory() as uow:
        prototype, = uow.prototypes.query()
    ws, artifacts = _workspace(tmp_path, sqlite_uow_factory)
    iface = DerivedInterfaceService(uow_factory=sqlite_uow_factory, artifacts=artifacts).create(
        prototype=prototype.uid_full, atoms=current_atoms_payload(),
    )
    with sqlite_uow_factory() as uow:
        ref = uow.artifacts.get_by_uid_full(iface.atoms_artifact_uid)
    ws.get_derived_interface_atoms(iface.uid_full)
    path = ws.artifact_path(ref)
    changed = json.loads(path.read_bytes())
    changed["scaled_positions"][0][0] += 0.25
    path.write_text(json.dumps(changed), encoding="utf-8")
    reopened, _ = _workspace(tmp_path, sqlite_uow_factory)
    with pytest.raises(ValueError, match="persisted identity"):
        reopened.get_derived_interface_atoms(iface.uid_full)
