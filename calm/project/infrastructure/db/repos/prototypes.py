"""Repository for authoritative interface prototypes."""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional, Sequence

from sqlalchemy import Connection, delete, insert, or_, select, update
from sqlalchemy.engine import RowMapping
from sqlalchemy.sql import func

from calm.analysis.pareto import (
    STRAIN_SIZE_PARETO_POLICY,
    STRAIN_SIZE_PARETO_POPULATION_SCOPE,
    STRAIN_SIZE_PARETO_VERSION,
    is_authoritative_pareto_metadata,
    is_strain_size_pareto_metadata,
)
from ....domain.models import Prototype, PrototypeSummary
from ....ports.repos import PrototypeRepository
from ..tables import edges as edges_t
from ..tables import prototypes as prototypes_t
from ..tables import runs as runs_t
from ..tables import slabs as slabs_t
from ._common import datetime_to_text as _dt_to_str
from .lineage import SqlAlchemyEdgeRepository


_RUN_SNAPSHOT_KIND = "prototype_run_snapshot"


class SqlAlchemyPrototypeRepository(PrototypeRepository):
    """Prototype persistence for Workspace v2.

    The canonical row describes the most recent observation of a scientific
    identity. Before changing its run, retain the outgoing observation in the
    existing edges table. Run-scoped queries recover that run's exact metrics
    and payload; unscoped queries continue to return one row per identity.
    """

    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    @staticmethod
    def _select_with_run_id_short() -> Any:
        """SELECT prototypes plus run_id_short (joined from runs).

        The prototypes table stores run_uid_full but not run_id_short. The
        Prototype domain model requires run_id_short, so we join against the
        runs table whenever we hydrate Prototype records.

        PrototypeSummary also surfaces the source slabs, so we also join
        slab_a/slab_b to provide their id_short values.
        """

        slab_a = slabs_t.alias("slab_a")
        slab_b = slabs_t.alias("slab_b")

        return select(
            prototypes_t,
            runs_t.c.id_short.label("run_id_short"),
            slab_a.c.id_short.label("slab_a_id_short"),
            slab_b.c.id_short.label("slab_b_id_short"),
        ).select_from(
            prototypes_t.outerjoin(
                runs_t, prototypes_t.c.run_uid_full == runs_t.c.uid_full
            )
            .outerjoin(slab_a, prototypes_t.c.slab_a_uid_full == slab_a.c.uid_full)
            .outerjoin(slab_b, prototypes_t.c.slab_b_uid_full == slab_b.c.uid_full)
        )

    def upsert_many(self, prototypes: Sequence[Prototype]) -> list[Prototype]:
        stored: list[Prototype] = []
        for proto in prototypes:
            existing = self.get_by_uid_full(proto.uid_full)
            if existing is not None and existing.run_uid_full != proto.run_uid_full:
                self._save_run_snapshot(existing)
            mutable_values = {
                "run_uid_full": proto.run_uid_full,
                "slab_a_uid_full": proto.slab_a_uid_full,
                "slab_b_uid_full": proto.slab_b_uid_full,
                "match_score": proto.match_score,
                "hencky_norm": proto.hencky_norm,
                "interface_area": proto.interface_area,
                "n_atoms": proto.natoms,
                "is_pareto": bool(proto.is_pareto),
                "pareto_rank": proto.pareto_rank,
                "payload_json": json.dumps(proto.payload) if proto.payload else None,
                "updated_at": func.current_timestamp(),
            }

            if existing is None:
                self._conn.execute(
                    insert(prototypes_t).values(
                        uid_full=proto.uid_full,
                        id_short=proto.id_short,
                        **mutable_values,
                    )
                )
            else:
                # Preserve the canonical identity, short ID, created_at, and PK.
                self._conn.execute(
                    update(prototypes_t)
                    .where(prototypes_t.c.uid_full == proto.uid_full)
                    .values(**mutable_values)
                )

            stmt = self._select_with_run_id_short().where(
                prototypes_t.c.uid_full == proto.uid_full
            )
            row = self._conn.execute(stmt).mappings().one()
            stored.append(self._row_to_prototype(row))

        return stored

    def _save_run_snapshot(self, proto: Prototype) -> None:
        """Keep one transactional snapshot per (run, scientific prototype).

        These are internal observations, separate from run_to_prototype lineage.
        Replacing an observation supports A -> B -> A -> B upserts, including
        returning to a previously stored payload, without stale hash ordering.
        No database schema migration is required.
        """
        self._conn.execute(
            delete(edges_t).where(
                edges_t.c.src_uid_full == proto.run_uid_full,
                edges_t.c.dst_uid_full == proto.uid_full,
                edges_t.c.kind == _RUN_SNAPSHOT_KIND,
            )
        )
        SqlAlchemyEdgeRepository(self._conn).add(
            src_uid_full=proto.run_uid_full,
            dst_uid_full=proto.uid_full,
            kind=_RUN_SNAPSHOT_KIND,
            payload={
                "run_uid_full": proto.run_uid_full,
                "run_id_short": proto.run_id_short,
                "match_score": proto.match_score,
                "hencky_norm": proto.hencky_norm,
                "interface_area": proto.interface_area,
                "n_atoms": proto.natoms,
                "is_pareto": proto.is_pareto,
                "pareto_rank": proto.pareto_rank,
                "payload_json": json.dumps(proto.payload),
                "created_at": proto.created_at,
            },
        )

    def _query_run_rows(
        self,
        run_uid_full: str,
        *,
        max_hencky_norm: float | None,
        min_match_score: float | None,
        order_by: str | None,
    ) -> list[dict[str, Any]]:
        snapshots = {
            edge.dst_uid_full: edge.payload
            for edge in SqlAlchemyEdgeRepository(self._conn).list(
                src_uid_full=run_uid_full, kind=_RUN_SNAPSHOT_KIND
            )
        }
        snapshot_uids = select(edges_t.c.dst_uid_full).where(
            edges_t.c.src_uid_full == run_uid_full,
            edges_t.c.kind == _RUN_SNAPSHOT_KIND,
        )
        stmt = self._select_with_run_id_short().where(
            or_(
                prototypes_t.c.run_uid_full == run_uid_full,
                prototypes_t.c.uid_full.in_(snapshot_uids),
            )
        ).order_by(prototypes_t.c.prototype_pk.asc())
        rows = []
        for stored in self._conn.execute(stmt).mappings().all():
            row = dict(stored)
            if row["run_uid_full"] != run_uid_full:
                row.update(snapshots[row["uid_full"]])
            # Filters and ordering must use the selected run's observation,
            # never the canonical row's metrics from a later search.
            if max_hencky_norm is not None and (
                row["hencky_norm"] is None or row["hencky_norm"] > max_hencky_norm
            ):
                continue
            if min_match_score is not None and (
                row["match_score"] is None or row["match_score"] < min_match_score
            ):
                continue
            row["created_at"] = _dt_to_str(row.get("created_at"))
            rows.append(row)
        if order_by:
            defaults = {
                "match_score": False,
                "hencky_norm": False,
                "interface_area": True,
                "n_atoms": False,
                "created_at": True,
            }
            key = order_by[1:] if order_by[0] in "+-" else order_by
            if key in defaults:
                descending = (
                    order_by.startswith("-")
                    if order_by[0] in "+-"
                    else defaults[key]
                )
                # SQLite sorts NULL first ascending, last descending.
                rows.sort(
                    key=lambda row: (row[key] is not None, row[key]),
                    reverse=descending,
                )
        return rows

    def get_by_uid_full(self, uid_full: str) -> Optional[Prototype]:
        row = (
            self._conn.execute(
                self._select_with_run_id_short().where(
                    prototypes_t.c.uid_full == uid_full
                )
            )
            .mappings()
            .first()
        )
        if row is None:
            return None
        return self._row_to_prototype(row)

    def get_many_by_uid_full(self, uid_fulls: Sequence[str]) -> list[Prototype]:
        if not uid_fulls:
            return []
        stmt = self._select_with_run_id_short().where(
            prototypes_t.c.uid_full.in_(list(uid_fulls))
        )
        rows = self._conn.execute(stmt).mappings().all()
        return [self._row_to_prototype(r) for r in rows]

    def query(  # noqa: C901
        self,
        *,
        run_uid_full: str | None = None,
        pareto_only: bool | None = None,
        max_hencky_norm: float | None = None,
        min_match_score: float | None = None,
        order_by: str | None = None,
        limit: int | None = None,
    ) -> list[PrototypeSummary]:
        if run_uid_full is not None:
            rows = self._query_run_rows(
                run_uid_full,
                max_hencky_norm=max_hencky_norm,
                min_match_score=min_match_score,
                order_by=order_by,
            )
            return self._summaries(rows, pareto_only=pareto_only, limit=limit)
        stmt = self._select_with_run_id_short()
        if max_hencky_norm is not None:
            stmt = stmt.where(prototypes_t.c.hencky_norm <= max_hencky_norm)
        if min_match_score is not None:
            stmt = stmt.where(prototypes_t.c.match_score >= min_match_score)

        # Ordering
        if order_by:
            key = order_by
            direction: str | None = None

            # Explicit direction prefixes.
            if order_by.startswith("-"):
                key = order_by[1:]
                direction = "desc"
            elif order_by.startswith("+"):
                key = order_by[1:]
                direction = "asc"

            col_map = {
                "match_score": prototypes_t.c.match_score,
                "hencky_norm": prototypes_t.c.hencky_norm,
                "interface_area": prototypes_t.c.interface_area,
                "n_atoms": prototypes_t.c.n_atoms,
                "created_at": prototypes_t.c.created_at,
            }
            col = col_map.get(key)
            if col is not None:
                # "Best-first" defaults (may be overridden by +/-).
                if direction is None:
                    direction = {
                        "match_score": "asc",  # Lower is better
                        "interface_area": "desc",
                        "hencky_norm": "asc",
                        "n_atoms": "asc",
                        "created_at": "desc",
                    }.get(key, "asc")

                stmt = stmt.order_by(col.desc() if direction == "desc" else col.asc())
        else:
            stmt = stmt.order_by(prototypes_t.c.prototype_pk.asc())

        if limit is not None and pareto_only is not True:
            stmt = stmt.limit(limit)

        rows = self._conn.execute(stmt).mappings().all()
        return self._summaries(rows, pareto_only=pareto_only, limit=limit)

    @classmethod
    def _summaries(
        cls,
        rows: Sequence[Mapping[str, Any]],
        *,
        pareto_only: bool | None,
        limit: int | None,
    ) -> list[PrototypeSummary]:
        summaries = [cls._row_to_prototype_summary(row) for row in rows]
        if pareto_only is True:
            summaries = [
                summary
                for summary in summaries
                if summary.is_pareto
                and summary.pareto_status == "authoritative"
                and summary.pareto_policy == STRAIN_SIZE_PARETO_POLICY
                and summary.pareto_policy_version == STRAIN_SIZE_PARETO_VERSION
                and summary.pareto_population_scope
                == STRAIN_SIZE_PARETO_POPULATION_SCOPE
            ]
        if limit is not None and limit >= 0:
            summaries = summaries[: int(limit)]
        return summaries

    @staticmethod
    def _row_to_prototype(row: RowMapping) -> Prototype:
        return Prototype(
            uid_full=row["uid_full"],
            id_short=row["id_short"],
            run_uid_full=row.get("run_uid_full"),
            run_id_short=row.get("run_id_short"),
            slab_a_uid_full=row.get("slab_a_uid_full"),
            slab_b_uid_full=row.get("slab_b_uid_full"),
            match_score=row.get("match_score"),
            hencky_norm=row.get("hencky_norm"),
            interface_area=row.get("interface_area"),
            natoms=row.get("n_atoms"),
            is_pareto=bool(row.get("is_pareto")),
            pareto_rank=row.get("pareto_rank"),
            payload=json.loads(row["payload_json"]) if row.get("payload_json") else {},
            created_at=_dt_to_str(row.get("created_at")),
        )

    @staticmethod
    def _summary_d_cell(
        payload: Mapping[str, Any],
        hencky_norm: Any,
    ) -> float | None:
        del hencky_norm
        metrics = payload.get("metrics")
        if not isinstance(metrics, Mapping) or metrics.get("d_cell") is None:
            return None
        return float(metrics["d_cell"])

    @staticmethod
    def _row_to_prototype_summary(row: RowMapping) -> PrototypeSummary:
        payload = json.loads(row["payload_json"]) if row.get("payload_json") else {}
        metadata = payload.get("pareto") if isinstance(payload, Mapping) else None
        valid_metadata = is_strain_size_pareto_metadata(metadata)
        if not valid_metadata:
            raise ValueError(
                "Current prototype records require complete strain-size Pareto metadata."
            )
        authoritative = is_authoritative_pareto_metadata(metadata)
        return PrototypeSummary(
            uid_full=row["uid_full"],
            id_short=row["id_short"],
            run_id_short=row.get("run_id_short"),
            run_uid_full=row.get("run_uid_full"),
            slab_a_uid_full=row.get("slab_a_uid_full"),
            slab_a_id_short=row.get("slab_a_id_short"),
            slab_b_uid_full=row.get("slab_b_uid_full"),
            slab_b_id_short=row.get("slab_b_id_short"),
            match_score=row.get("match_score"),
            hencky_norm=row.get("hencky_norm"),
            interface_area=row.get("interface_area"),
            natoms=row.get("n_atoms"),
            d_cell=SqlAlchemyPrototypeRepository._summary_d_cell(
                payload, row.get("hencky_norm")
            ),
            is_pareto=(bool(metadata.get("is_member")) if valid_metadata else False),
            pareto_rank=(metadata.get("rank") if valid_metadata else None),
            pareto_policy=(metadata.get("policy") if valid_metadata else None),
            pareto_policy_version=(metadata.get("version") if valid_metadata else None),
            pareto_population_scope=(
                metadata.get("population_scope") if valid_metadata else None
            ),
            pareto_population_size=(
                metadata.get("population_size") if valid_metadata else None
            ),
            pareto_d_cell_key=(metadata.get("d_cell_key") if valid_metadata else None),
            pareto_status=("authoritative" if authoritative else "scoped_alternative"),
            payload=dict(payload),
            created_at=_dt_to_str(row.get("created_at")),
        )
