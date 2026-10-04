"""Unlimited public lists and health counts must pass the first 50 records."""

from slab_record_fixtures import current_slab_payload, current_slab_uid
from calm.project.application.bulks import BulksService
from calm.project.domain.models import Slab
from calm.project.infrastructure.artifacts.fs_store import FSArtifactStore
from calm.project.runtime.workspace import Workspace
from calm.public.persistence.adapter import WorkspaceAdapter
from calm.public.persistence.repository import PublicRepository
from calm.public.queries.health import ProjectHealthService
from calm.public.queries.project import PublicProjectQueries


def test_unlimited_queries_collections_and_health_counts(tmp_path, sqlite_uow_factory):
    service = BulksService(uow_factory=sqlite_uow_factory)
    bulks = [service.add_bulk(label=f"bulk {i}", payload={"index": i}) for i in range(60)]
    with sqlite_uow_factory() as uow:
        for bulk in bulks:
            payload = current_slab_payload(bulk_uid_full=bulk.uid_full)
            uid = current_slab_uid(bulk_uid_full=bulk.uid_full, miller=(1, 0, 0), payload=payload)
            uow.slabs.create_many([Slab(
                uid_full=uid, id_short=uow.ids.ensure_short_id(uid_full=uid, tag="s"),
                bulk_uid_full=bulk.uid_full, bulk_id_short=bulk.id_short,
                miller=(1, 0, 0), payload=payload,
            )])
    workspace = Workspace(
        out_dir=tmp_path / "out", artifact_store=FSArtifactStore(tmp_path / "out"),
        uow_factory=sqlite_uow_factory,
    )
    assert len(workspace.list_bulks()) == 60
    assert len(workspace.list_slabs()) == 60
    assert len(workspace.list_slabs(bulk=bulks[-1].uid_full)) == 1
    for limit, expected in ((None, 60), (0, 0), (7, 7), (100, 60)):
        assert len(workspace.list_bulks(limit=limit)) == expected
        assert len(workspace.list_slabs(limit=limit)) == expected

    adapter = WorkspaceAdapter(workspace)
    repository = PublicRepository(adapter)
    queries = PublicProjectQueries(adapter, repo=repository, project=None)
    assert len(queries.materials()) == 60
    health = ProjectHealthService(
        project=None, workspace=workspace, repository=repository, queries=queries,
    )
    counts = health.project_counts()
    assert counts["materials"] == counts["surfaces"] == 60
