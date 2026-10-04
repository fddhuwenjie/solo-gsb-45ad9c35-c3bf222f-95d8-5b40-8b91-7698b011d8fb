"""Cross-pool allocation migration: move in-use subnets between pools.

``POST /pools/{id}/allocation-migrations`` moves a caller-named manifest of
still-allocated subnets from the source pool to a target pool while keeping
every CIDR, ``request_id`` and allocation row identity (row id,
``prefix_length``, ``created_at``) intact.

The whole operation -- idempotency, both pools' state pinning, every
validation, the ownership update and the migration record -- runs inside one
``BEGIN IMMEDIATE`` write transaction, so a failed validation changes nothing
in either pool and concurrent writers are serialised.

Each manifest item must currently belong to the source pool with exactly the
submitted CIDR; the target pool must completely contain every moved CIDR, and
no moved CIDR may overlap the target pool's exclusions or its remaining
allocations.  The migration is refused when either pool has a ``planned`` or
``active`` BGP announcement plan, so no pinned plan summary can drift from the
pool state it was created against.

``operation_id`` is the idempotency key: retrying with the same parameters
(source pool, target pool and the request_id-ordered manifest) replays the
original result; the same key with different parameters is a 409 conflict.
"""

from __future__ import annotations

import json
import sqlite3

from .cidrmath import parse_network
from .db import write_transaction
from .service import (
    ApiError,
    _pool_allocations,
    _pool_exclusions,
    _row_pool,
    utc_now,
)


def _migration_error(detail: str, code: str, status_code: int = 422) -> ApiError:
    return ApiError(status_code, detail, code=code)


def _canonical_params(
    source_pool_id: int,
    target_pool_id: int,
    parsed: list[tuple[str, object]],
) -> str:
    """Build the parameter fingerprint of a migration request.

    The manifest is processed in input order and the mapping is returned in
    input order, so order is significant: replaying with a different order is
    a different migration.  CIDRs are stored in canonical (strict-parsed)
    form, so only host-bit/notation differences are normalised away.
    """

    params = {
        "source_pool_id": source_pool_id,
        "target_pool_id": target_pool_id,
        "allocations": [[request_id, str(net)] for request_id, net in parsed],
    }
    return json.dumps(params, separators=(",", ":"))


def _response(
    operation_id: str,
    migration_id: int,
    source_pool_id: int,
    target_pool_id: int,
    allocations: list[dict],
    created_at: str,
    replayed: bool,
) -> dict:
    return {
        "operation_id": operation_id,
        "migration_id": migration_id,
        "source_pool_id": source_pool_id,
        "target_pool_id": target_pool_id,
        "allocations": allocations,
        "created_at": created_at,
        "replayed": replayed,
    }


def migrate_allocations(
    source_pool_id: int,
    operation_id: str,
    target_pool_id: int,
    items,
) -> dict:
    """Move the manifest allocations from the source pool to the target pool.

    Everything runs inside one immediate write transaction; any raised error
    rolls the whole migration back and leaves both pools untouched.  ``items``
    is a list of pydantic objects exposing ``request_id`` and ``cidr``.
    """

    # Cheap, state-independent checks run before the transaction opens:
    # strict CIDR parsing and manifest-internal duplicate detection never
    # need the database, and canonical CIDRs are what the fingerprint stores.
    parsed: list[tuple[str, object]] = []
    seen_requests: set[str] = set()
    for item in items:
        try:
            net = parse_network(item.cidr)
        except ValueError as exc:
            raise _migration_error(
                f"invalid allocation CIDR {item.cidr!r}: {exc}", "invalid_cidr"
            )
        if item.request_id in seen_requests:
            raise _migration_error(
                f"request_id {item.request_id!r} appears more than once in the "
                "migration manifest",
                "migration_duplicate_request_id",
            )
        seen_requests.add(item.request_id)
        parsed.append((item.request_id, net))

    params_json = _canonical_params(source_pool_id, target_pool_id, parsed)

    with write_transaction() as conn:
        # Idempotency first, as for partitions: a matching record with the
        # same fingerprint replays the original result; any parameter change
        # is a conflict.  This wins over every other check.
        previous = conn.execute(
            "SELECT * FROM allocation_migrations WHERE operation_id = ?",
            (operation_id,),
        ).fetchone()
        if previous is not None:
            if previous["params"] != params_json:
                raise ApiError(
                    409,
                    f"operation_id {operation_id!r} was already used with "
                    "different parameters",
                    code="operation_id_conflict",
                )
            return _response(
                previous["operation_id"],
                previous["id"],
                previous["source_pool_id"],
                previous["target_pool_id"],
                json.loads(previous["result"]),
                previous["created_at"],
                replayed=True,
            )

        # Pin both pools' state inside the write transaction.
        _row_pool(conn, source_pool_id)
        if target_pool_id == source_pool_id:
            raise _migration_error(
                f"target pool {target_pool_id} is the same as the source pool",
                "migration_same_pool",
            )
        target_row = conn.execute(
            "SELECT * FROM pools WHERE id = ?",
            (target_pool_id,),
        ).fetchone()
        if target_row is None:
            raise ApiError(
                404,
                f"target pool {target_pool_id} not found",
                code="migration_target_pool_not_found",
            )
        target_net = parse_network(target_row["cidr"])

        # Refuse while either pool has a pinned BGP announcement plan
        # (planned or active): moving allocations would desynchronise the
        # plan summary and coverage from the pool state it was created
        # against.
        for pool_id in (source_pool_id, target_pool_id):
            blocking = conn.execute(
                "SELECT plan_id, status FROM bgp_plans WHERE pool_id = ? "
                "AND status IN ('planned', 'active') ORDER BY plan_id LIMIT 1",
                (pool_id,),
            ).fetchone()
            if blocking is not None:
                raise ApiError(
                    409,
                    f"pool {pool_id} has a {blocking['status']} BGP announcement "
                    f"plan {blocking['plan_id']!r}; allocation migration would "
                    "invalidate its pinned pool-state summary",
                    code="migration_blocked_by_bgp_plan",
                )

        source_rows = {
            row["request_id"]: row for row in _pool_allocations(conn, source_pool_id)
        }
        target_exclusions = [
            parse_network(row["cidr"]) for row in _pool_exclusions(conn, target_pool_id)
        ]
        target_rows = _pool_allocations(conn, target_pool_id)
        target_nets = [(parse_network(row["cidr"]), row) for row in target_rows]

        moves: list[tuple[sqlite3.Row, object]] = []
        for request_id, net in parsed:
            row = source_rows.get(request_id)
            if row is None:
                raise _migration_error(
                    f"request_id {request_id!r} is not currently allocated in "
                    f"source pool {source_pool_id}",
                    "migration_allocation_not_in_source",
                )
            if parse_network(row["cidr"]) != net:
                raise _migration_error(
                    f"request_id {request_id!r} currently owns {row['cidr']}, "
                    f"not the submitted {net}",
                    "migration_cidr_mismatch",
                )
            if not net.subnet_of(target_net):
                raise _migration_error(
                    f"allocation {net} (request_id {request_id!r}) is not "
                    f"completely inside target pool {target_net}",
                    "migration_outside_target",
                )
            blocking_excl = next(
                (excluded for excluded in target_exclusions if net.overlaps(excluded)),
                None,
            )
            if blocking_excl is not None:
                raise ApiError(
                    409,
                    f"allocation {net} (request_id {request_id!r}) overlaps "
                    f"exclusion {blocking_excl} in target pool {target_pool_id}",
                    code="migration_target_exclusion_overlap",
                )
            blocking_alloc = next(
                (other for other_net, other in target_nets if net.overlaps(other_net)),
                None,
            )
            if blocking_alloc is not None:
                raise ApiError(
                    409,
                    f"allocation {net} (request_id {request_id!r}) overlaps "
                    f"target pool allocation {blocking_alloc['cidr']} "
                    f"(request_id {blocking_alloc['request_id']!r})",
                    code="migration_target_allocation_overlap",
                )
            moves.append((row, net))

        # All checks passed: flip ownership while preserving every row's
        # identity (id, cidr, request_id, prefix_length, created_at).
        now = utc_now()
        moved: list[dict] = []
        try:
            for row, net in moves:
                conn.execute(
                    "UPDATE allocations SET pool_id = ? WHERE id = ?",
                    (target_pool_id, row["id"]),
                )
                moved.append(
                    {
                        "request_id": row["request_id"],
                        "prefix_length": row["prefix_length"],
                        "cidr": str(net),
                        "allocation_id": row["id"],
                        "replayed": False,
                    }
                )

            cursor = conn.execute(
                "INSERT INTO allocation_migrations "
                "(operation_id, source_pool_id, target_pool_id, params, result, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    operation_id,
                    source_pool_id,
                    target_pool_id,
                    params_json,
                    json.dumps(moved),
                    now,
                ),
            )
            migration_id = cursor.lastrowid
        except sqlite3.IntegrityError:
            raise ApiError(
                409,
                f"migration {operation_id!r} conflicts with existing data",
                code="migration_conflict",
            )

    return _response(
        operation_id,
        migration_id,
        source_pool_id,
        target_pool_id,
        moved,
        now,
        replayed=False,
    )
