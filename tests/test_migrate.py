"""Cross-pool allocation migration: move in-use subnets between pools."""

from __future__ import annotations

import threading

from fastapi.testclient import TestClient


def _create_pool(client, cidr, exclusions=None):
    resp = client.post(
        "/pools", json={"cidr": cidr, "exclusions": exclusions or []}
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def _allocate(client, pool_id, request_id, prefix_length):
    resp = client.post(
        f"/pools/{pool_id}/allocations",
        json={"request_id": request_id, "prefix_length": prefix_length},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _migrate(client, source_id, operation_id, target_id, allocations):
    return client.post(
        f"/pools/{source_id}/allocation-migrations",
        json={
            "operation_id": operation_id,
            "target_pool_id": target_id,
            "allocations": allocations,
        },
    )


def _status(client, pool_id):
    resp = client.get(f"/pools/{pool_id}")
    assert resp.status_code == 200, resp.text
    return resp.json()


def _two_pools(client, source_cidr="10.0.0.0/24", source_exclusions=None,
               target_cidr="10.0.0.0/16", target_exclusions=None):
    src = _create_pool(client, source_cidr, source_exclusions)
    dst = _create_pool(client, target_cidr, target_exclusions)
    return src, dst


def test_migration_moves_allocations_preserving_identity(client):
    src, dst = _two_pools(client, source_exclusions=["10.0.0.192/26"])
    a = _allocate(client, src, "req-a", 26)  # 10.0.0.0/26
    b = _allocate(client, src, "req-b", 26)  # 10.0.0.64/26
    c = _allocate(client, src, "req-c", 26)  # 10.0.0.128/26

    # Manifest deliberately out of address order: the response must echo
    # the submitted input order.
    resp = _migrate(
        client, src, "op-1", dst,
        [
            {"request_id": "req-c", "cidr": c["cidr"]},
            {"request_id": "req-a", "cidr": a["cidr"]},
        ],
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["operation_id"] == "op-1"
    assert body["source_pool_id"] == src
    assert body["target_pool_id"] == dst
    assert body["replayed"] is False
    assert isinstance(body["migration_id"], int)
    assert [item["request_id"] for item in body["allocations"]] == ["req-c", "req-a"]
    assert [item["cidr"] for item in body["allocations"]] == [
        "10.0.0.128/26",
        "10.0.0.0/26",
    ]
    assert [item["allocation_id"] for item in body["allocations"]] == [c["id"], a["id"]]
    assert [item["prefix_length"] for item in body["allocations"]] == [26, 26]
    assert all(item["replayed"] is False for item in body["allocations"])

    # Ownership flipped: source keeps only req-b, target owns the moved pair.
    assert [x["request_id"] for x in _status(client, src)["allocated"]] == ["req-b"]
    moved = _status(client, dst)["allocated"]
    assert [x["request_id"] for x in moved] == ["req-a", "req-c"]
    assert {x["cidr"] for x in moved} == {"10.0.0.0/26", "10.0.0.128/26"}

    # request_id idempotency now follows the allocation to the target pool,
    # and the row id is unchanged.
    replay = client.post(
        f"/pools/{dst}/allocations",
        json={"request_id": "req-a", "prefix_length": 26},
    )
    assert replay.status_code == 201
    assert replay.json()["replayed"] is True
    assert replay.json()["id"] == a["id"]
    assert replay.json()["cidr"] == a["cidr"]

    # The same request_id no longer resolves against the source pool.
    stale = client.post(
        f"/pools/{src}/allocations",
        json={"request_id": "req-a", "prefix_length": 26},
    )
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "request_id_conflict"

    # Release now targets the new owner.
    freed = client.delete(f"/pools/{dst}/allocations/req-c")
    assert freed.status_code == 200
    assert freed.json()["released"] == c["cidr"]
    assert freed.json()["pool_id"] == dst


def test_migration_then_back_again(client):
    src, dst = _two_pools(client)
    a = _allocate(client, src, "req-a", 26)

    assert _migrate(
        client, src, "op-forward", dst, [{"request_id": "req-a", "cidr": a["cidr"]}]
    ).status_code == 201
    back = _migrate(
        client, dst, "op-back", src, [{"request_id": "req-a", "cidr": a["cidr"]}]
    )
    assert back.status_code == 201, back.text
    assert [x["request_id"] for x in _status(client, src)["allocated"]] == ["req-a"]
    assert _status(client, dst)["allocated"] == []
    # Row identity survived both moves.
    assert back.json()["allocations"][0]["allocation_id"] == a["id"]


def test_migration_replay_returns_original_result(client):
    src, dst = _two_pools(client)
    a = _allocate(client, src, "req-a", 26)
    manifest = [{"request_id": "req-a", "cidr": a["cidr"]}]

    first = _migrate(client, src, "op-idem", dst, manifest)
    assert first.status_code == 201, first.text

    second = _migrate(client, src, "op-idem", dst, manifest)
    assert second.status_code == 200, second.text
    body = second.json()
    assert body["replayed"] is True
    assert body["migration_id"] == first.json()["migration_id"]
    # The stored item mapping is returned verbatim (its replayed flag records
    # the original move, like batch reservations); the top-level flag marks
    # this response as a replay.
    assert body["allocations"] == first.json()["allocations"]
    assert all(item["replayed"] is False for item in body["allocations"])
    assert body["source_pool_id"] == src
    assert body["target_pool_id"] == dst

    # The replay did not move anything a second time and created no extra data.
    assert [x["request_id"] for x in _status(client, dst)["allocated"]] == ["req-a"]
    assert _status(client, src)["allocated"] == []


def test_migration_replay_works_after_request_id_released(client):
    # The stored result is returned verbatim even when current pool state no
    # longer matches the original manifest.
    src, dst = _two_pools(client)
    a = _allocate(client, src, "req-a", 26)
    manifest = [{"request_id": "req-a", "cidr": a["cidr"]}]
    assert _migrate(client, src, "op-rel", dst, manifest).status_code == 201
    assert client.delete(f"/pools/{dst}/allocations/req-a").status_code == 200

    replay = _migrate(client, src, "op-rel", dst, manifest)
    assert replay.status_code == 200
    assert replay.json()["replayed"] is True
    assert replay.json()["allocations"][0]["cidr"] == a["cidr"]
    # The replay did not resurrect the released allocation.
    assert _status(client, dst)["allocated"] == []


def test_migration_conflict_on_changed_target(client):
    src, dst = _two_pools(client)
    other = _create_pool(client, "10.1.0.0/16")
    a = _allocate(client, src, "req-a", 26)

    ok = _migrate(client, src, "op-x", dst, [{"request_id": "req-a", "cidr": a["cidr"]}])
    assert ok.status_code == 201

    # Same operation_id against the other pool after moving the allocation
    # back to the source.
    assert _migrate(client, dst, "op-back", src, [{"request_id": "req-a", "cidr": a["cidr"]}]).status_code == 201
    resp = _migrate(client, src, "op-x", other, [{"request_id": "req-a", "cidr": a["cidr"]}])
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "operation_id_conflict"


def test_migration_conflict_on_changed_manifest(client):
    src, dst = _two_pools(client)
    a = _allocate(client, src, "req-a", 26)
    b = _allocate(client, src, "req-b", 26)

    assert _migrate(
        client, src, "op-y", dst, [{"request_id": "req-a", "cidr": a["cidr"]}]
    ).status_code == 201

    # Different manifest (added item), same operation_id -> conflict.
    resp = _migrate(
        client, src, "op-y", dst,
        [
            {"request_id": "req-a", "cidr": a["cidr"]},
            {"request_id": "req-b", "cidr": b["cidr"]},
        ],
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "operation_id_conflict"

    # Order is part of the parameters too.
    resp = _migrate(
        client, src, "op-y", dst,
        [
            {"request_id": "req-b", "cidr": b["cidr"]},
            {"request_id": "req-a", "cidr": a["cidr"]},
        ],
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "operation_id_conflict"

    # Neither conflict attempt moved anything: req-b is still in the source.
    assert [x["request_id"] for x in _status(client, src)["allocated"]] == ["req-b"]


def test_migration_duplicate_request_id_rejected(client):
    src, dst = _two_pools(client)
    a = _allocate(client, src, "req-a", 26)
    before = _status(client, src)

    resp = _migrate(
        client, src, "op-dup", dst,
        [
            {"request_id": "req-a", "cidr": a["cidr"]},
            {"request_id": "req-a", "cidr": a["cidr"]},
        ],
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "migration_duplicate_request_id"
    assert _status(client, src) == before


def test_migration_allocation_not_in_source_rejected(client):
    src, dst = _two_pools(client)
    other = _create_pool(client, "10.2.0.0/16")
    a = _allocate(client, other, "req-other", 26)  # 10.2.0.0/26

    # Belongs to a third pool.
    resp = _migrate(
        client, src, "op-missing", dst,
        [{"request_id": "req-other", "cidr": a["cidr"]}],
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "migration_allocation_not_in_source"

    # Unknown request_id altogether.
    resp = _migrate(
        client, src, "op-missing2", dst,
        [{"request_id": "nope", "cidr": "10.0.0.0/26"}],
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "migration_allocation_not_in_source"


def test_migration_cidr_mismatch_rejected(client):
    src, dst = _two_pools(client)
    _allocate(client, src, "req-a", 26)  # 10.0.0.0/26
    before = _status(client, src)

    resp = _migrate(
        client, src, "op-mismatch", dst,
        [{"request_id": "req-a", "cidr": "10.0.0.64/26"}],
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "migration_cidr_mismatch"
    assert _status(client, src) == before
    assert _status(client, dst)["allocated"] == []


def test_migration_invalid_cidr_rejected(client):
    src, dst = _two_pools(client)
    _allocate(client, src, "req-a", 26)

    for bad in ("10.0.0.0/40", "10.0.0.5/26", "not-a-cidr", "::1/128"):
        resp = _migrate(
            client, src, f"op-bad-{bad}", dst,
            [{"request_id": "req-a", "cidr": bad}],
        )
        assert resp.status_code == 422, (bad, resp.text)
        assert resp.json()["error"]["code"] == "invalid_cidr", bad
    assert [x["request_id"] for x in _status(client, src)["allocated"]] == ["req-a"]


def test_migration_target_must_contain_cidr(client):
    src = _create_pool(client, "10.0.0.0/24")
    dst = _create_pool(client, "10.1.0.0/16")
    a = _allocate(client, src, "req-a", 26)

    resp = _migrate(
        client, src, "op-outside", dst,
        [{"request_id": "req-a", "cidr": a["cidr"]}],
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "migration_outside_target"
    # Nothing moved.
    assert [x["request_id"] for x in _status(client, src)["allocated"]] == ["req-a"]
    assert _status(client, dst)["allocated"] == []


def test_migration_same_pool_rejected(client):
    src, _ = _two_pools(client)
    a = _allocate(client, src, "req-a", 26)
    resp = _migrate(
        client, src, "op-self", src,
        [{"request_id": "req-a", "cidr": a["cidr"]}],
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "migration_same_pool"


def test_migration_source_pool_not_found(client):
    dst = _create_pool(client, "10.0.0.0/16")
    resp = _migrate(
        client, 999, "op-404", dst,
        [{"request_id": "req-a", "cidr": "10.0.0.0/26"}],
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "pool_not_found"


def test_migration_target_pool_not_found(client):
    src = _create_pool(client, "10.0.0.0/24")
    a = _allocate(client, src, "req-a", 26)
    resp = _migrate(
        client, src, "op-t404", 999,
        [{"request_id": "req-a", "cidr": a["cidr"]}],
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "migration_target_pool_not_found"
    assert [x["request_id"] for x in _status(client, src)["allocated"]] == ["req-a"]


def test_migration_overlaps_target_exclusion_rejected(client):
    src, dst = _two_pools(
        client,
        target_cidr="10.0.0.0/16",
        target_exclusions=["10.0.0.0/24"],
    )
    a = _allocate(client, src, "req-a", 26)  # 10.0.0.0/26 inside the target exclusion
    before_src = _status(client, src)
    before_dst = _status(client, dst)

    resp = _migrate(
        client, src, "op-excl", dst,
        [{"request_id": "req-a", "cidr": a["cidr"]}],
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "migration_target_exclusion_overlap"
    assert _status(client, src) == before_src
    assert _status(client, dst) == before_dst


def test_migration_overlaps_target_allocation_rejected(client):
    src, dst = _two_pools(client)
    a = _allocate(client, src, "req-a", 25)  # 10.0.0.0/25
    # Occupy the exact same CIDR in the target via a normal allocation.
    t = client.post(
        f"/pools/{dst}/allocations",
        json={"request_id": "t-1", "prefix_length": 25},
    )
    assert t.status_code == 201, t.text
    before_src = _status(client, src)
    before_dst = _status(client, dst)

    resp = _migrate(
        client, src, "op-ovl", dst,
        [{"request_id": "req-a", "cidr": a["cidr"]}],
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "migration_target_allocation_overlap"
    assert _status(client, src) == before_src
    assert _status(client, dst) == before_dst


def test_migration_partial_overlap_rolls_back_entire_manifest(client):
    # Two items, first valid, second hitting a target exclusion: neither
    # allocation may move -- the failure is atomic for both pools.
    src, dst = _two_pools(
        client,
        target_cidr="10.0.0.0/16",
        target_exclusions=["10.0.0.128/26"],
    )
    a = _allocate(client, src, "req-a", 26)  # 10.0.0.0/26 - free in target
    b = _allocate(client, src, "req-b", 26)  # 10.0.0.64/26
    c = _allocate(client, src, "req-c", 26)  # 10.0.0.128/26 - excluded in target
    before_src = _status(client, src)
    before_dst = _status(client, dst)

    resp = _migrate(
        client, src, "op-atomic", dst,
        [
            {"request_id": "req-a", "cidr": a["cidr"]},
            {"request_id": "req-c", "cidr": c["cidr"]},
        ],
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "migration_target_exclusion_overlap"
    assert _status(client, src) == before_src
    assert _status(client, dst) == before_dst

    # The failed operation_id is not consumed: the corrected manifest can
    # reuse it.
    ok = _migrate(
        client, src, "op-atomic", dst,
        [
            {"request_id": "req-b", "cidr": b["cidr"]},
            {"request_id": "req-a", "cidr": a["cidr"]},
        ],
    )
    assert ok.status_code == 201, ok.text
    assert [x["request_id"] for x in ok.json()["allocations"]] == ["req-b", "req-a"]
    assert [x["request_id"] for x in _status(client, src)["allocated"]] == ["req-c"]


def test_migration_blocked_by_planned_plan_on_source(client):
    src = _create_pool(client, "203.0.112.0/24")
    dst = _create_pool(client, "203.0.112.0/23")
    _allocate(client, src, "order-1", 25)
    plan = {
        "plan_id": "plan-src",
        "relationships": [
            {"type": "customer_provider", "customer": 10, "provider": 100}
        ],
        "announcements": [
            {"id": "a0", "prefix": "203.0.112.0/25", "as_path": [10, 100]},
            {"id": "a1", "prefix": "203.0.112.128/25", "as_path": [10, 100]},
        ],
    }
    assert client.post(f"/pools/{src}/bgp-plans", json=plan).status_code == 201

    resp = _migrate(
        client, src, "op-blocked", dst,
        [{"request_id": "order-1", "cidr": "203.0.112.0/25"}],
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "migration_blocked_by_bgp_plan"
    assert [x["request_id"] for x in _status(client, src)["allocated"]] == ["order-1"]
    assert _status(client, dst)["allocated"] == []


def test_migration_blocked_by_active_plan_on_target(client):
    src = _create_pool(client, "203.0.118.0/24")
    dst = _create_pool(client, "203.0.118.0/23")
    # Plan pinned against the target pool (which has no allocations yet).
    plan = {
        "plan_id": "plan-dst",
        "relationships": [
            {"type": "customer_provider", "customer": 10, "provider": 100}
        ],
        "announcements": [
            {"id": "a0", "prefix": "203.0.118.0/25", "as_path": [10, 100]},
            {"id": "a1", "prefix": "203.0.118.128/25", "as_path": [10, 100]},
        ],
    }
    assert client.post(f"/pools/{dst}/bgp-plans", json=plan).status_code == 201
    assert client.post(
        f"/pools/{dst}/bgp-plans/plan-dst/activate"
    ).status_code == 200

    _allocate(client, src, "order-1", 25)  # 203.0.118.0/25
    resp = _migrate(
        client, src, "op-blocked2", dst,
        [{"request_id": "order-1", "cidr": "203.0.118.0/25"}],
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "migration_blocked_by_bgp_plan"
    assert [x["request_id"] for x in _status(client, src)["allocated"]] == ["order-1"]


def test_migration_allowed_after_plan_rechecks_still_hold(client):
    # Sanity companion: with no plan on either pool, the migration proceeds.
    src = _create_pool(client, "10.9.0.0/24")
    dst = _create_pool(client, "10.9.0.0/23")
    a = _allocate(client, src, "req-a", 25)
    resp = _migrate(
        client, src, "op-ok", dst,
        [{"request_id": "req-a", "cidr": a["cidr"]}],
    )
    assert resp.status_code == 201, resp.text


def test_concurrent_migrations_same_operation_id_replay(client):
    src = _create_pool(client, "10.0.0.0/24")
    dst = _create_pool(client, "10.0.0.0/16")
    cidrs = []
    for i in range(4):
        cidrs.append(_allocate(client, src, f"req-{i}", 26)["cidr"])
    manifest = [{"request_id": f"req-{i}", "cidr": cidrs[i]} for i in range(4)]

    bodies: list[dict] = []
    barrier = threading.Barrier(6)

    from app.main import app

    def worker():
        local = TestClient(app)
        barrier.wait()
        resp = _migrate(local, src, "op-conc", dst, manifest)
        assert resp.status_code in (200, 201), resp.text
        bodies.append(resp.json())

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(1 for b in bodies if not b["replayed"]) == 1
    assert sum(1 for b in bodies if b["replayed"]) == 5
    assert {b["migration_id"] for b in bodies} == {bodies[0]["migration_id"]}
    assert len(_status(client, dst)["allocated"]) == 4
    assert _status(client, src)["allocated"] == []


def test_concurrent_migrations_are_serialised(client):
    # Two different operations moving two allocations into the same target
    # CIDR space: the write transaction serialises them; they cannot both
    # succeed against overlapping space, and a successful one commits fully.
    src1 = _create_pool(client, "10.10.0.0/24")
    src2 = _create_pool(client, "10.11.0.0/24")
    dst = _create_pool(client, "10.0.0.0/8")
    a = _allocate(client, src1, "a-1", 25)  # 10.10.0.0/25
    b = _allocate(client, src2, "b-1", 25)  # 10.11.0.0/25 (disjoint -> both fine)
    results: list[int] = []
    barrier = threading.Barrier(2)

    from app.main import app

    def worker(source_id, op, request_id, cidr):
        local = TestClient(app)
        barrier.wait()
        resp = _migrate(
            local, source_id, op, dst,
            [{"request_id": request_id, "cidr": cidr}],
        )
        results.append(resp.status_code)

    threads = [
        threading.Thread(target=worker, args=(src1, "op-a", "a-1", a["cidr"])),
        threading.Thread(target=worker, args=(src2, "op-b", "b-1", b["cidr"])),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(results) == [201, 201]
    moved = {x["request_id"] for x in _status(client, dst)["allocated"]}
    assert moved == {"a-1", "b-1"}
    assert _status(client, src1)["allocated"] == []
    assert _status(client, src2)["allocated"] == []
