from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, insert

from calm.analysis.pareto import authoritative_pareto_metadata
from calm.project.domain.models import Prototype
from calm.project.infrastructure.db.repos import SqlAlchemyPrototypeRepository
from calm.project.infrastructure.db.tables import bulks, metadata, runs, slabs
from calm.public.persistence.adapter import WorkspaceAdapter
from calm.public.persistence.repository import PublicRepository


@pytest.fixture
def database(tmp_path):
    url = f"sqlite:///{tmp_path / 'observations.sqlite'}"
    engine = create_engine(url)
    metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(insert(bulks).values(
            bulk_pk=1, uid_full="bulk:test", id_short="b_test", payload_json="{}",
        ))
        connection.execute(insert(slabs).values(
            uid_full="slab:test", id_short="s_test", bulk_pk=1, payload_json="{}",
        ))
        connection.execute(insert(runs), [
            dict(uid_full=f"run:{name}", id_short=f"r_{name}", run_type="prototype_search", status="done", spec_json="{}")
            for name in ("a", "b")
        ])
    try:
        yield engine, url
    finally:
        engine.dispose()


def _prototype(run, *, uid="proto:shared", score=0.4, member=True):
    return Prototype(
        uid_full=uid, id_short="p_" + uid.split(":")[-1],
        run_uid_full=f"run:{run}", run_id_short=f"r_{run}",
        slab_a_uid_full="slab:test", slab_b_uid_full="slab:test",
        match_score=score, hencky_norm=score / 2, interface_area=4.0,
        natoms=4, is_pareto=member, pareto_rank=0 if member else None,
        payload={
            "source": run, "metrics": {"d_cell": score},
            "pareto": authoritative_pareto_metadata(
                is_member=member, rank=0 if member else None,
                population_size=2, d_cell_key=int(score * 1e12),
            ),
        },
    )


def test_shared_identity_keeps_both_search_observations_after_reopen(database):
    engine, url = database
    first = _prototype("a")
    second = _prototype("b", score=0.7, member=False)
    with engine.begin() as connection:
        repo = SqlAlchemyPrototypeRepository(connection)
        repo.upsert_many([first])
        repo.upsert_many([replace(second, id_short="p_ignored")])
    engine.dispose()
    reopened = create_engine(url)
    try:
        with reopened.begin() as connection:
            repo = SqlAlchemyPrototypeRepository(connection)
            a, = repo.query(run_uid_full="run:a")
            b, = repo.query(run_uid_full="run:b")
            assert a.uid_full == b.uid_full == first.uid_full
            assert a.id_short == b.id_short == first.id_short
            assert (a.run_id_short, b.run_id_short) == ("r_a", "r_b")
            assert (a.match_score, b.match_score) == (0.4, 0.7)
            assert a.payload == first.payload
            assert b.payload == second.payload
            assert a.is_pareto and not b.is_pareto
            assert len(repo.query()) == 1  # Scientific identity remains unique.

            searches = {name: SimpleNamespace(name=name, run_uid_full=f"run:{name}", search_identity=name, spec={}) for name in ("a", "b")}
            workspace = SimpleNamespace(
                list_prototypes=lambda *, run, limit: repo.query(run_uid_full=run, limit=limit),
                get_interface_search=lambda name: searches[name],
            )
            public = PublicRepository(WorkspaceAdapter(workspace))
            for name, score in (("a", 0.4), ("b", 0.7)):
                row, = public.list_candidates(search_name=name)
                assert row["search_name"] == name
                assert row["score"] == score
    finally:
        reopened.dispose()


def test_run_filters_ordering_and_limit_use_the_original_metrics(database):
    engine, _ = database
    with engine.begin() as connection:
        repo = SqlAlchemyPrototypeRepository(connection)
        repo.upsert_many([_prototype("a"), _prototype("a", uid="proto:other", score=0.5)])
        repo.upsert_many([_prototype("b", score=0.7, member=False)])
        assert [row.uid_full for row in repo.query(run_uid_full="run:a", order_by="match_score", limit=1)] == ["proto:shared"]
        assert [row.uid_full for row in repo.query(run_uid_full="run:a", order_by="-match_score", limit=1)] == ["proto:other"]
        assert [row.uid_full for row in repo.query(run_uid_full="run:a", max_hencky_norm=0.21)] == ["proto:shared"]
        assert [row.uid_full for row in repo.query(run_uid_full="run:a", min_match_score=0.45)] == ["proto:other"]
        assert len(repo.query(run_uid_full="run:a", pareto_only=True)) == 2
        assert not repo.query(run_uid_full="run:b", pareto_only=True)
        assert not repo.query(run_uid_full="run:a", limit=0)


def test_revisiting_runs_updates_only_the_selected_observation(database):
    engine, _ = database
    with engine.begin() as connection:
        repo = SqlAlchemyPrototypeRepository(connection)
        for run, score in (("a", 0.4), ("b", 0.7), ("a", 0.6), ("b", 0.8), ("a", 0.4), ("b", 0.7)):
            repo.upsert_many([_prototype(run, score=score)])
        a, = repo.query(run_uid_full="run:a")
        b, = repo.query(run_uid_full="run:b")
        assert (a.match_score, b.match_score) == (0.4, 0.7)


def test_failed_upsert_rolls_back_snapshot_and_canonical_row(database):
    engine, _ = database
    with engine.begin() as connection:
        SqlAlchemyPrototypeRepository(connection).upsert_many([_prototype("a")])
    with pytest.raises(RuntimeError, match="abort"):
        with engine.begin() as connection:
            SqlAlchemyPrototypeRepository(connection).upsert_many([_prototype("b")])
            raise RuntimeError("abort")
    with engine.begin() as connection:
        repo = SqlAlchemyPrototypeRepository(connection)
        assert len(repo.query(run_uid_full="run:a")) == 1
        assert not repo.query(run_uid_full="run:b")
