from __future__ import annotations

import pytest
from sqlalchemy import create_engine, insert

from calm.project.infrastructure.db.id_resolver import SqlAlchemyIdResolver
from calm.project.infrastructure.db.tables import metadata, runs


# Distinct valid UIDs whose short-ID digests share their first eight digits.
UID_A = "proto:v2:e5539085c50b2c0b11b0912395a5add6620109fa48bf6136513ba3c4a49cb775"
UID_B = "proto:v2:f48fc0238feaa512d860af618387fbdb0826027cca390473bfc82ec5a18773bf"


def _insert_run(connection, uid, short):
    # The resolver hashes the full UID independently of the table's kind.
    connection.execute(insert(runs).values(
        uid_full=uid, id_short=short, run_type="test", status="done", spec_json="{}",
    ))


def test_batch_allocations_reserve_colliding_short_ids():
    engine = create_engine("sqlite:///:memory:")
    metadata.create_all(engine)
    try:
        with engine.begin() as connection:
            ids = SqlAlchemyIdResolver(connection)
            first = ids.ensure_short_id(tag="r", uid_full=UID_A)
            second = ids.ensure_short_id(tag="r", uid_full=UID_B)
            assert first == "r_3dcd9aec"
            assert second == "r_3dcd9aec6"
            assert ids.ensure_short_id(tag="r", uid_full=UID_B) == second
            _insert_run(connection, UID_A, first)
            _insert_run(connection, UID_B, second)
    finally:
        engine.dispose()


def test_exact_short_ids_remain_resolvable_after_a_collision():
    engine = create_engine("sqlite:///:memory:")
    metadata.create_all(engine)
    try:
        with engine.begin() as connection:
            ids = SqlAlchemyIdResolver(connection)
            first = ids.ensure_short_id(tag="r", uid_full=UID_A)
            _insert_run(connection, UID_A, first)
            second = ids.ensure_short_id(tag="r", uid_full=UID_B)
            _insert_run(connection, UID_B, second)
        with engine.begin() as connection:
            ids = SqlAlchemyIdResolver(connection)
            assert ids.resolve(first, expected_tag="r") == UID_A
            assert ids.resolve(second, expected_tag="r") == UID_B
            with pytest.raises(ValueError, match="ambiguous"):
                ids.resolve("r_3dcd", expected_tag="r")
            with pytest.raises(ValueError, match="Expected"):
                ids.resolve(first, expected_tag="p")
            # Existing IDs must not shrink when allocation bounds change.
            assert ids.ensure_short_id(
                tag="r", uid_full=UID_B, min_prefix_len=4,
            ) == second
    finally:
        engine.dispose()


def test_new_unit_of_work_does_not_retain_rolled_back_reservations():
    from calm.project.infrastructure.db.uow import SqlAlchemyUnitOfWork

    engine = create_engine("sqlite:///:memory:")
    metadata.create_all(engine)
    uow = SqlAlchemyUnitOfWork(engine)
    try:
        with pytest.raises(RuntimeError, match="abort"):
            with uow as active:
                assert active.ids.ensure_short_id(tag="r", uid_full=UID_A) == "r_3dcd9aec"
                raise RuntimeError("abort")
        with uow as active:
            assert active.ids.ensure_short_id(tag="r", uid_full=UID_B) == "r_3dcd9aec"
    finally:
        engine.dispose()
