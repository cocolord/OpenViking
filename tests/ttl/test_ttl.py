import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import fakeredis.aioredis
import pytest
import pytest_asyncio

from openviking.ttl.adapter import TTLBackfillRequest, TTLRecycleRequest
from openviking.ttl.runtime import DirectoryTTL, next_check_ms
from openviking_cli.utils.config.ttl_config import TTLConfig


@pytest.fixture
def engine(monkeypatch, caplog):
    from openviking.ttl.runtime import logger

    monkeypatch.setattr(logger, "handlers", [caplog.handler])
    e = DirectoryTTL(
        resource_id="r", timezone_name="UTC", stat_rate=100000, delete_interval=0, batch_interval=0
    )
    e.redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    e.pair = (TTLConfig(sessions=7, user_events=7, peer_events=7), None)

    async def policies(_):
        return e.pair

    e.policies = policies
    e.fs = SimpleNamespace(
        ls=AsyncMock(return_value=[]),
        stat=AsyncMock(),
        rm=AsyncMock(),
        _uri_to_path=lambda uri, **kw: uri,
        _async_agfs=SimpleNamespace(
            pathlock_acquire_tree=AsyncMock(return_value={}),
            pathlock_release=AsyncMock(),
        ),
    )
    e.account_manager = SimpleNamespace(
        get_account=AsyncMock(return_value={"status": "active"}),
        list_accounts=AsyncMock(return_value=[]),
        list_users_page=AsyncMock(return_value={"users": []}),
    )
    return e


@pytest.fixture
def scheduler(engine):
    scheduler = SimpleNamespace(register_handler=Mock(), report=AsyncMock())
    engine.install_expiry(scheduler)
    return scheduler


@pytest.mark.asyncio
async def test_config_reset_intent_precedes_save_and_disable_skips_redis(engine, scheduler):
    e = engine
    e.pair = (TTLConfig(sessions="disabled"), None)
    e.manager = SimpleNamespace(patch_account=AsyncMock())

    async def save(*args):
        state = await e.read_initial("a")
        assert state["reset_required"]["viking://user/_ttl_/sessions"]
        scheduler.report.assert_awaited_once()

    e.manager.patch_account.side_effect = save
    from openviking_cli.exceptions import UnavailableError

    scheduler.report.side_effect = ConnectionError("scheduler unavailable")
    with pytest.raises(UnavailableError):
        await e.patch_configuration({"ttl": {"sessions": 7}}, "a")
    e.manager.patch_account.assert_not_called()
    scheduler.report.reset_mock(side_effect=True)
    await e.patch_configuration({"ttl": {"sessions": 7}}, "a")
    redis = e.redis
    e.client = AsyncMock(side_effect=ConnectionError("Redis unavailable"))
    e.manager.patch_account.reset_mock()
    with pytest.raises(UnavailableError):
        await e.patch_configuration({"ttl": {"sessions": 7}}, "a")
    e.manager.patch_account.assert_not_called()
    e.manager.patch_account.side_effect = None
    await e.patch_configuration({"ttl": {"sessions": "disabled"}}, "a")
    e.manager.patch_account.assert_awaited_once()
    assert await redis.hlen(e.prefix("a") + "backfill:reset_required") > 0
    scheduler.report.assert_awaited_once()


def test_hourly_window_is_stable_and_skips_missed_rounds():
    now = int(datetime(2026, 10, 20, 12, 34, 56, tzinfo=timezone.utc).timestamp() * 1000)
    first = next_check_ms("ov-one", now)
    assert now < first <= now + 3_600_000
    assert next_check_ms("ov-one", first) == first + 3_600_000
    assert next_check_ms("ov-one", now) == first
    assert next_check_ms("ov-two", now) != first


@pytest.mark.asyncio
async def test_libraries_defer_discovery_after_enable_reset_and_restart(monkeypatch):
    from openviking.scheduling.expiry.api import CallbackRequest

    clock = int(datetime(2026, 10, 28, tzinfo=timezone.utc).timestamp() * 1000)
    monkeypatch.setattr("openviking.ttl.runtime.time.time_ns", lambda: clock * 1_000_000)
    server = fakeredis.FakeServer()
    root = "viking://user/u/sessions"
    pair = (TTLConfig(sessions=7), None)

    def replacement(resource):
        e = DirectoryTTL(resource_id=resource, batch_interval=0, stat_rate=100000)
        e.redis = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
        e.manager = SimpleNamespace(resolve_account=AsyncMock(return_value=pair))
        tree = {"viking://user": ["viking://user/u"], root: [root + "/old"]}
        e.fs = SimpleNamespace(
            ls=AsyncMock(
                side_effect=lambda uri, **kw: [
                    {"uri": child, "isDir": True} for child in tree.get(uri, [])
                ]
            ),
            stat=AsyncMock(return_value={"modTime": "2099-01-01T00:00:00Z"}),
            read_file=AsyncMock(return_value="{}"),
        )
        e.account_manager = SimpleNamespace(
            get_account=AsyncMock(return_value={"status": "active"}),
            list_users_page=AsyncMock(return_value={"users": []}),
        )
        e.install_expiry(SimpleNamespace(register_handler=Mock(), report=AsyncMock()))
        return e

    engines = []
    try:
        for resource in ("ov-one", "ov-two"):
            old = replacement(resource)
            try:
                if resource == "ov-one":
                    old.manager.resolve_account.return_value = (None, None)
                    old.manager.patch_account = AsyncMock()
                    await old.patch_configuration({"ttl": {"sessions": 7}}, "a")
                else:
                    prefix = old.prefix("a") + "backfill:"
                    await old.redis.hset(prefix + "initial", "roots", json.dumps([root]))
                    await old.redis.set(prefix + "initial:done", "1")
                    await old.register_initial_backfill(
                        TTLBackfillRequest(resource, "a", reset=True)
                    )
                old.fs.ls.assert_not_called()
            finally:
                await old.aclose()
            e = replacement(resource)
            engines.append(e)
            await e.start()
            await e.policy("a", root)  # First business access after a simultaneous release.
            await e.register_initial_backfill(TTLBackfillRequest(resource, "a"))
            await e.on_post_write(root + "/active/messages.jsonl", e.context("a"), clock)
            await e._registration_task
            assert await e.redis.zscore(e.prefix("a") + "sessions", root + "/active") == clock
            await e.handle(CallbackRequest(e._identity("a")))  # Overdue callback from the old Pod.
            assert e._task is None
            e.fs.ls.assert_not_called()
            e.fs.stat.assert_not_called()

        assert len({e._recycle_after_ms for e in engines}) == 2
        for e in sorted(engines, key=lambda e: e._recycle_after_ms):
            clock = e._recycle_after_ms
            await e.handle(CallbackRequest(e._identity("a")))
            async with asyncio.timeout(2):
                await e._task
            state = await e.read_initial("a")
            assert state["done"] and not state["discover"]
            e.fs.stat.assert_awaited_once_with(root + "/old/messages.jsonl", ctx=e.context("a"))
            for later in engines:
                if later._recycle_after_ms > clock:
                    later.fs.ls.assert_not_called()
    finally:
        for e in engines:
            await e.aclose()


@pytest.mark.asyncio
async def test_incremental_keeps_buffer_until_schedule_is_durable(engine, scheduler):
    e = engine
    uri = "viking://user/u/sessions/s"
    await e.on_post_write(uri + "/messages.jsonl", e.context("a"), 1000)
    scheduler.report.side_effect = ConnectionError("scheduler unavailable")
    with pytest.raises(ConnectionError):
        await e._retry_writes()
    assert e._retry == {("a", "sessions", uri): 1000}
    scheduler.report.side_effect = None
    await e._retry_writes()
    await e.on_post_write(uri + "/messages.jsonl", e.context("a"), 2000)
    await e._retry_writes()
    assert not e._retry and scheduler.report.await_count == 2
    assert await e.redis.zscore(e.prefix("a") + "sessions", uri) == 2000


@pytest.mark.asyncio
@pytest.mark.parametrize("done", [False, True])
async def test_expiry_restart_resumes_checkpoint_without_full_scan(
    engine, scheduler, monkeypatch, done
):
    from openviking.scheduling.expiry.api import CallbackRequest, CallbackStatus

    e = engine
    root = "viking://user/u/sessions"
    prefix = e.prefix("a") + "backfill:"
    # Persisted state from a previous writer. Its in-memory queues are gone.
    checkpoint = {"root": 0, "stack": [{"uri": root, "last": root + "/old"}]}
    await e.redis.hset(
        prefix + "initial",
        mapping={
            "roots": json.dumps([root]),
            "checkpoint": json.dumps(checkpoint),
        },
    )
    if done:
        await e.redis.set(prefix + "initial:done", "1")
    else:
        await e.redis.sadd(prefix + "pending", "initial")
    e.fs.ls.return_value = [{"uri": root + "/old", "isDir": True}]
    await e.start()
    request = CallbackRequest(e._identity("a"))
    try:
        e.manager = SimpleNamespace(resolve_account=AsyncMock(return_value=e.pair))
        await DirectoryTTL.policies(e, "a")
        await e.policy("a", root)
        await e.publish_user_configuration(e.context("a"), None)
        await e.register_initial_backfill(TTLBackfillRequest("r", "a"))
        assert e._task is None
        e.fs.ls.assert_not_called()
        scheduler.report.side_effect = ConnectionError("Redis down")
        with pytest.raises(ConnectionError):
            await e.handle(request)
        assert not e._recycle and e._task is None
        scheduler.report.side_effect = None
        # An overdue task at restart only persists the next staggered wake-up.
        result = await e.handle(request)
        assert result.status is CallbackStatus.COMPLETED and e._task is None
        e.fs.ls.assert_not_called()
        e.account_manager.list_accounts.assert_not_called()
        monkeypatch.setattr(
            "openviking.ttl.runtime.time.time_ns", lambda: e._recycle_after_ms * 1_000_000
        )
        await e.handle(request)
        async with asyncio.timeout(2):
            await e._task
        assert (await e.read_initial("a"))["done"]
        assert e.fs.ls.await_count == (0 if done else 1)
        if not done:
            assert e.fs.ls.call_args.args == (root,)
        e.fs.stat.assert_not_called()  # The checkpoint excludes the old child.
        # Disable retains the hourly check without resuming scans or deletion.
        e.pair = (None, None)
        e.fs.ls.reset_mock()
        await e.handle(request)
        async with asyncio.timeout(2):
            await e._task
        assert not e._pending and not e._recycle
        e.fs.ls.assert_not_called()
        e.fs.stat.assert_not_called()
        e.fs.rm.assert_not_called()
    finally:
        await e.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("peer_days", [None, "disabled", 7])
async def test_initial_discovers_roots_then_resumes_without_listing(engine, peer_days):
    e = engine
    e.pair = (TTLConfig(sessions=7, peer_events=peer_days), None)
    e._cache_user_policy(("a", "other"), TTLConfig(peer_events=7))
    tree = {
        "viking://user": ["viking://user/u", "viking://user/other"],
        "viking://user/u/peers": ["viking://user/u/peers/p"],
        "viking://user/other/peers": [
            "viking://user/other/peers/p1",
            "viking://user/other/peers/p2",
        ],
    }
    e.fs.ls.side_effect = lambda uri, **kw: [{"uri": u, "isDir": True} for u in tree.get(uri, [])]
    await e.register_initial_backfill(TTLBackfillRequest("r", "a"))
    assert (await e.read_initial("a"))["discover"]
    e.fs.ls.assert_not_called()
    await e._resume_initial("a")
    state = await e.read_initial("a")
    assert not state["discover"]
    roots = json.loads(state["roots"])
    assert "viking://user/u/sessions" in roots
    assert all(root + "/memories/events" in roots for root in tree["viking://user/other/peers"])
    assert ("viking://user/u/peers/p/memories/events" in roots) == (peer_days == 7)
    assert e.fs.ls.await_count == (3 if peer_days == 7 else 2)
    assert all(call.kwargs["node_limit"] is None for call in e.fs.ls.call_args_list)
    e.fs.ls.reset_mock()
    await e.register_initial_backfill(TTLBackfillRequest("r", "a"))
    e.fs.ls.assert_not_called()
    await e.redis.set(e.prefix("a") + "backfill:initial:done", "1")
    await e.recycle_due(TTLRecycleRequest("r", "a"))
    e.fs.ls.assert_not_called()


@pytest.mark.asyncio
async def test_hourly_trigger_recovers_intent_with_old_done(engine):
    e = engine
    root = "viking://user/u/sessions"
    e.pair = (TTLConfig(sessions=7), None)
    await e.register_initial_backfill(TTLBackfillRequest("r", "a", (root,)))
    prefix = e.prefix("a") + "backfill:"
    await e.redis.set(prefix + "initial:done", "1")
    await e.redis.hset(prefix + "reset_required", root, "saved-token")
    await e.recycle_due(TTLRecycleRequest("r", "a"))
    state = await e.read_initial("a")
    assert not state["done"] and state["pending"] and state["discover"]
    assert state["reset_required"]
    e.fs.ls.assert_not_called()
    await e._resume_initial("a")
    assert not (await e.read_initial("a"))["reset_required"]
    e.fs.ls.reset_mock()
    await e.register_initial_backfill(TTLBackfillRequest("r", "a"))
    e.fs.ls.assert_not_called()


@pytest.mark.asyncio
async def test_write_scores_retry_and_expiry(engine):
    e = engine
    ctx = e.context("a")
    uri = "viking://user/u/sessions/s"
    for now in (2000, 1000, 2000):
        await e.on_post_write(uri + "/messages.jsonl", ctx, now)
    assert await e.redis.zscore(e.prefix("a") + "sessions", uri) is None
    await e._retry_writes()
    assert await e.redis.zscore(e.prefix("a") + "sessions", uri) == 2000
    e.fs.stat.assert_not_called()
    event = "viking://user/u/memories/events/2020/01/01"
    for now in (2000, 99999):
        await e.on_post_write(event + "/file.md", ctx, now)
    await e._retry_writes()
    score = await e.redis.zscore(e.prefix("a") + "user_events", event)
    assert score == 1577836800000
    echo = await e.resolve_expires_at(
        [uri, uri + "/archive/file", event, event + "/a", "viking://resources/a"], ctx
    )
    assert echo[uri] == echo[uri + "/archive/file"] == "1970-01-08T00:00:02Z"
    assert echo[event] == echo[event + "/a"] == "2020-01-08T00:00:00Z"
    assert echo["viking://resources/a"] is None
    register = e.redis.zadd
    e.redis.zadd = AsyncMock(side_effect=ConnectionError("redis down"))
    await e.on_post_write(uri + "/messages.jsonl", ctx, 5000)
    e.redis.zadd.assert_not_called()
    with pytest.raises(ConnectionError):
        await e._retry_writes()
    assert e._retry[("a", "sessions", uri)] == 5000
    e.redis.zadd = register
    await e._retry_writes()
    assert not e._retry
    assert await e.redis.zscore(e.prefix("a") + "sessions", uri) == 5000


@pytest.mark.asyncio
async def test_disabled_account_cached_policy_no_business_io(engine):
    e = engine
    e.pair = (TTLConfig(sessions=7), TTLConfig(sessions="disabled"))
    e._cache_user_policy(("a", "u"), None)
    e.client = AsyncMock(side_effect=AssertionError("off write path touched Redis"))
    e._dispatcher.run = AsyncMock(side_effect=AssertionError("cached off path dispatched work"))
    uri = "viking://user/u/sessions/s"
    await e.on_post_write(uri + "/messages.jsonl", e.context("a"), 1)
    assert await e.resolve_expires_at([uri], e.context("a")) == {uri: None}
    await e.check_target(uri + "/messages.jsonl", e.context("a"))
    e.client.assert_not_called()
    e._dispatcher.run.assert_not_called()
    assert not e._retry and not e._wake.is_set()
    e.fs.stat.assert_not_called()
    e.fs._async_agfs.pathlock_acquire_tree.assert_not_called()


@pytest.mark.asyncio
async def test_expiry_echo_tracks_shared_peer_policy_without_filesystem_io(engine, monkeypatch):
    from openviking.server.identity import RequestContext, Role
    from openviking.ttl import wiring
    from openviking_cli.session.user_id import UserIdentifier

    e = engine
    e.writer = False  # Reader echoes policies without starting maintenance work.
    e.pair = (TTLConfig(sessions=30, peer_events=30), TTLConfig(sessions=14, peer_events=14))
    ctx = RequestContext(user=UserIdentifier("a", "u"), role=Role.ROOT)
    monkeypatch.setattr(wiring, "_adapter", e)
    session = "viking://user/u/sessions/one"
    peers = [f"viking://user/u/peers/{p}/memories/events/2026/01/01" for p in ("p1", "p2")]
    user_event = "viking://user/u/memories/events/2026/01/01"
    missing = "viking://user/u/sessions/unregistered"
    await e.redis.zadd(e.prefix("a") + "sessions", {session: 1767225600000})
    e.redis.zmscore = Mock(wraps=e.redis.zmscore)
    e.fs.read_file = AsyncMock(side_effect=AssertionError("echo read a cached UserConfig"))
    for policy, session_date, peer_date in (
        (TTLConfig(sessions=3, peer_events=7, user_events=60), "01-04", "01-08"),
        (TTLConfig(sessions=5, peer_events=2, user_events=60), "01-06", "01-03"),
        (TTLConfig(sessions="disabled", peer_events="disabled", user_events=60), None, None),
        (TTLConfig(user_events=60), "01-15", "01-15"),  # Removed overrides inherit Account.
    ):
        await e.publish_user_configuration(ctx, policy)
        e.redis.zmscore.reset_mock()
        entries = [
            {"uri": uri}
            for uri in [session, session + "/messages.jsonl", *peers, user_event, missing]
        ]
        await wiring.attach_expires_at(entries, ctx)
        expected = [
            f"2026-{session_date}T00:00:00Z" if session_date else None,
            f"2026-{session_date}T00:00:00Z" if session_date else None,
            f"2026-{peer_date}T00:00:00Z" if peer_date else None,
            f"2026-{peer_date}T00:00:00Z" if peer_date else None,
            "2026-03-02T00:00:00Z",
            None,
        ]
        assert [entry["expires_at"] for entry in entries] == expected
        if session_date:
            e.redis.zmscore.assert_called_once_with(e.prefix("a") + "sessions", [session, missing])
        else:
            e.redis.zmscore.assert_not_called()
    e.fs.stat.assert_not_called()
    e.fs.ls.assert_not_called()
    e.fs.read_file.assert_not_called()
    assert e._task is None and not e._retry


def test_event_dates_follow_writer_timezone(monkeypatch):
    import time

    from openviking.ttl.runtime import directory

    item = directory("viking://user/u/memories/events/2026/03/08")
    try:
        with monkeypatch.context() as env:
            env.setenv("TZ", "America/New_York")
            time.tzset()
            local = DirectoryTTL(resource_id="r")
            explicit = DirectoryTTL(resource_id="r", timezone_name="America/New_York")
            assert local.base(item) == explicit.base(item) == 1772946000000
            assert local.expiry(item, local.base(item), 1) == explicit.expiry(
                item, explicit.base(item), 1
            )
            # Calendar days retain local midnight across the DST transition.
            assert local.expiry(item, local.base(item), 1) - local.base(item) == 23 * 3600 * 1000
    finally:
        time.tzset()


@pytest.mark.asyncio
async def test_initial_partial_failure_reset_and_no_rescan(engine):
    e = engine
    root = "viking://user/u/sessions"
    uris = [root + "/a", root + "/b"]
    e.fs.ls.return_value = [{"uri": u, "isDir": True} for u in uris]
    e.fs.stat.return_value = {"modTime": "2020-01-01T00:00:00Z"}
    await e.redis.zadd(e.prefix("a") + "sessions", {uris[0]: 123})
    request = TTLBackfillRequest("r", "a", (root,))
    await e.register_initial_backfill(request)
    original = e.redis.zadd
    e.redis.zadd = AsyncMock(side_effect=ConnectionError("injected"))
    prefix = e.prefix("a") + "backfill:"
    before = await e.redis.hget(prefix + "initial", "checkpoint")
    with pytest.raises(ConnectionError):
        await e._scan("a")
    assert await e.redis.hget(prefix + "initial", "checkpoint") == before
    assert await e.redis.get(prefix + "initial:done") is None
    e.redis.zadd = original
    await e._scan("a")
    assert await e.redis.get(prefix + "initial:done") == "1"
    assert await e.redis.zscore(e.prefix("a") + "sessions", uris[0]) == 123
    assert all(call.args[0] == uris[1] + "/messages.jsonl" for call in e.fs.stat.call_args_list)
    e.fs.ls.reset_mock()
    await e.register_initial_backfill(request)
    await e._scan("a")
    e.fs.ls.assert_not_called()
    await e.redis.zrem(e.prefix("a") + "sessions", uris[1])
    await e.register_initial_backfill(TTLBackfillRequest("r", "a", (root,), reset=True))
    assert await e.redis.get(prefix + "initial:done") is None
    await e._scan("a")
    assert await e.redis.get(prefix + "initial:done") == "1"
    assert await e.redis.zscore(e.prefix("a") + "sessions", uris[1]) == 1577836800000
    assert all([await e.redis.ttl(key) == -1 for key in await e.redis.keys("*")])


@pytest.mark.asyncio
@pytest.mark.parametrize("with_registry", [False, True])
async def test_session_renewal_and_delete_retry(engine, with_registry):
    e = engine
    if not with_registry:
        e.account_manager = None  # Local dev mode still supports TTL.
    uri = "viking://user/u/sessions/s"
    key = e.prefix("a") + "sessions"
    await e.redis.set(e.prefix("a") + "backfill:initial:done", "1")
    await e.redis.zadd(key, {uri: 1})
    now = datetime.now(timezone.utc)
    e.fs.stat.return_value = {"modTime": now.isoformat()}
    result = await e.recycle_due(TTLRecycleRequest("r", "a"))
    assert result.skipped >= 1
    e.fs.rm.assert_not_called()
    assert await e.redis.zscore(key, uri) > 1
    await e.redis.zadd(key, {uri: 1})
    e.fs.stat.return_value = {"modTime": "2000-01-01T00:00:00Z"}
    e.fs.rm.side_effect = OSError("delete failed")
    result = await e.recycle_due(TTLRecycleRequest("r", "a"))
    assert result.failed == 1 and await e.redis.zscore(key, uri) == 1
    e.fs.rm.side_effect = None
    result = await e.recycle_due(TTLRecycleRequest("r", "a"))
    assert result.deleted == 1 and await e.redis.zscore(key, uri) is None
    await e.redis.zadd(key, {uri: 1})
    e.fs.stat.side_effect = FileNotFoundError(uri)
    e.fs.rm.side_effect = FileNotFoundError(uri)
    result = await e.recycle_due(TTLRecycleRequest("r", "a"))
    assert result.deleted == 1 and await e.redis.zscore(key, uri) is None


@pytest.mark.asyncio
async def test_shutdown_finishes_inflight_delete_before_closing_redis(engine):
    e = engine
    uri = "viking://user/u/sessions/s"
    await e.redis.set(e.prefix("a") + "backfill:initial:done", "1")
    await e.redis.zadd(e.prefix("a") + "sessions", {uri: 1})
    e.fs.stat.return_value = {"modTime": "2000-01-01T00:00:00Z"}
    entered, release = asyncio.Event(), asyncio.Event()

    async def deleting(*args, **kwargs):
        entered.set()
        await release.wait()

    e.fs.rm.side_effect = deleting
    redis = e.redis
    redis.aclose = AsyncMock(wraps=redis.aclose)
    recycle = asyncio.create_task(e.recycle_due(TTLRecycleRequest("r", "a", 2**53 - 1)))
    await asyncio.wait_for(entered.wait(), timeout=1)
    closing = asyncio.create_task(e.aclose())
    await asyncio.sleep(0)
    assert not closing.done()
    redis.aclose.assert_not_called()
    release.set()
    assert (await recycle).deleted == 1
    await closing
    redis.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_reset_waits_inflight_scan(engine):
    e = engine
    root = "viking://user/u/sessions"
    entered, release = asyncio.Event(), asyncio.Event()

    async def listing(*args, **kwargs):
        entered.set()
        await release.wait()
        return []

    e.fs.ls.side_effect = listing
    await e.register_initial_backfill(TTLBackfillRequest("r", "a", (root,)))

    async def scan():
        async with e._locks["a"]:
            await e._scan("a")

    old = asyncio.create_task(scan())
    await entered.wait()
    resetting = asyncio.create_task(
        e.register_initial_backfill(TTLBackfillRequest("r", "a", (root,), True))
    )
    await asyncio.sleep(0)
    assert not resetting.done()
    release.set()
    await old
    await resetting
    assert await e.redis.get(e.prefix("a") + "backfill:initial:done") is None
    assert await e.redis.sismember(e.prefix("a") + "backfill:pending", "initial")
    await e._scan("a")
    assert await e.redis.get(e.prefix("a") + "backfill:initial:done") == "1"


@pytest.mark.asyncio
async def test_missing_session_echo_does_not_stat(engine):
    e = engine
    uri = "viking://user/u/sessions/missing"
    assert await e.resolve_expires_at([uri], e.context("a")) == {uri: None}
    e.fs.stat.assert_not_called()


@pytest.mark.asyncio
async def test_date_scan_never_lists_day_or_session_archive(engine):
    e = engine
    root = "viking://user/u/memories/events"
    date = root + "/2020/01/01"
    directories = {
        root: [root + "/2020"],
        root + "/2020": [root + "/2020/01"],
        root + "/2020/01": [date],
    }

    async def listing(uri, **kw):
        assert uri in directories
        return [{"uri": value, "isDir": True} for value in directories[uri]]

    e.fs.ls.side_effect = listing
    await e.register_initial_backfill(TTLBackfillRequest("r", "a", (root,)))
    await e._scan("a")
    assert await e.redis.zscore(e.prefix("a") + "user_events", date) == 1577836800000
    e.fs.stat.assert_not_called()


@pytest_asyncio.fixture
async def reset_case(engine):
    root = "viking://user/u/sessions"
    prefix = engine.prefix("a") + "backfill:"
    await engine.redis.set(prefix + "initial:done", "1")
    await engine.redis.hset(prefix + "reset_required", root, "token")
    request = TTLBackfillRequest("r", "a", (root,), reset=True, reset_intents={root: "token"})
    return engine, root, prefix, request


@pytest.mark.asyncio
@pytest.mark.parametrize("discover", [False, True])
async def test_reset_intent_replay_keeps_new_checkpoint_and_newer_intent(reset_case, discover):
    e, root, prefix, request = reset_case
    if discover:
        request = TTLBackfillRequest("r", "a", reset=True, reset_intents=request.reset_intents)
    await e.redis.zadd(e.prefix("a") + "sessions", {root + "/kept": 123})
    await e.register_initial_backfill(request)
    if discover:
        assert (await e.read_initial("a"))["discover"]
        assert await e.redis.hget(prefix + "reset_required", root) == "token"
        e.fs.ls.assert_not_called()
        e._initial_roots = AsyncMock(return_value=[root])
        await e._resume_initial("a")
    assert await e.redis.get(prefix + "initial:done") is None
    assert await e.redis.hget(prefix + "reset_required", root) is None
    assert await e.redis.sismember(prefix + "pending", "initial")
    assert await e.redis.zscore(e.prefix("a") + "sessions", root + "/kept") == 123
    # The acknowledgement was lost. Neither a replay nor a late old request
    # may erase progress or consume a newer configuration's intent.
    await e.redis.hset(prefix + "initial", "checkpoint", "new-progress")
    await e.redis.set(prefix + "initial:done", "1")
    await e.register_initial_backfill(request)
    await e.redis.hset(prefix + "reset_required", root, "second")
    await e.register_initial_backfill(request)
    assert await e.redis.hget(prefix + "initial", "checkpoint") == "new-progress"
    assert await e.redis.get(prefix + "initial:done") == "1"
    assert await e.redis.hget(prefix + "reset_required", root) == "second"
    e.fs.ls.assert_not_called()


@pytest.mark.asyncio
async def test_reset_intent_redis_failure_then_retry(reset_case):
    e, root, prefix, request = reset_case
    original = e.redis.eval
    e.redis.eval = AsyncMock(side_effect=ConnectionError("before commit"))
    from openviking_cli.exceptions import UnavailableError

    with pytest.raises(UnavailableError):
        await e.register_initial_backfill(request)
    assert await e.redis.hget(prefix + "reset_required", root) == "token"
    assert await e.redis.get(prefix + "initial:done") == "1"
    e.redis.eval = original
    await e.register_initial_backfill(request)
    assert await e.redis.hget(prefix + "reset_required", root) is None
    assert await e.redis.get(prefix + "initial:done") is None
    e.fs.ls.return_value = [{"uri": root + "/missing", "isDir": True}]
    e.fs.stat.return_value = {"modTime": "2020-01-01T00:00:00Z"}
    await e._scan("a")
    assert await e.redis.zscore(e.prefix("a") + "sessions", root + "/missing") == 1577836800000
    assert await e.redis.get(prefix + "initial:done") == "1"


@pytest.mark.asyncio
async def test_disabled_reset_intent_does_not_scan_or_acknowledge(reset_case):
    e, root, prefix, request = reset_case
    e.pair = (TTLConfig(user_events=7, sessions="disabled"), None)
    await e.register_initial_backfill(request)
    assert await e.redis.hget(prefix + "reset_required", root) == "token"
    assert await e.redis.get(prefix + "initial:done") == "1"
    e.fs.ls.assert_not_called()


@pytest.mark.asyncio
async def test_async_batch_grouping_partial_failure_and_concurrent_update(engine):
    e = engine
    session = "viking://user/u/sessions/s"
    event = "viking://user/u/memories/events/2020/01/01"
    for uri, now in (
        (session + "/messages.jsonl", 1),
        (event + "/a", 1),
        (event + "/b", 2),
    ):
        await e.on_post_write(uri, e.context("a"), now)
    await e.on_post_write(session + "/messages.jsonl", e.context("b"), 1)
    original = e.redis.zadd

    async def register(key, entries, **options):
        if key == e.prefix("a") + "sessions":
            await e.on_post_write(session + "/messages.jsonl", e.context("a"), 3)
        if key == e.prefix("b") + "sessions":
            raise ConnectionError("one group failed")
        return await original(key, entries, **options)

    e.redis.zadd = AsyncMock(side_effect=register)
    with pytest.raises(ConnectionError):
        await e._retry_writes()
    assert e.redis.zadd.await_count == 3
    assert e.redis.zadd.call_args_list[1].kwargs == {"nx": True}
    assert len(e.redis.zadd.call_args_list[1].args[1]) == 1
    assert e._retry == {("a", "sessions", session): 3, ("b", "sessions", session): 1}
    e.redis.zadd = original
    await e._retry_writes()
    assert not e._retry
    assert await e.redis.zscore(e.prefix("a") + "sessions", session) == 3


@pytest.mark.asyncio
async def test_failed_head_batch_does_not_starve_later_registration(engine):
    e = engine
    e.batch_size = 1
    bad, good = ["viking://user/u/sessions/" + name for name in ("bad", "good")]
    await e.on_post_write(bad + "/messages.jsonl", e.context("a"), 1)
    await e.on_post_write(good + "/messages.jsonl", e.context("a"), 2)
    original = e.redis.zadd

    async def register(key, entries, **options):
        if bad in entries:
            raise ConnectionError("first directory failed")
        return await original(key, entries, **options)

    e.redis.zadd = AsyncMock(side_effect=register)
    with pytest.raises(ConnectionError):
        await e._retry_writes()
    await e._retry_writes()
    assert list(e._retry) == [("a", "sessions", bad)]
    assert await e.redis.zscore(e.prefix("a") + "sessions", good) == 2


@pytest.mark.asyncio
async def test_async_buffer_full_merges_existing_and_reports_drop(engine, caplog):
    e = engine
    e.retry_limit = 1
    first, second = ["viking://user/u/sessions/" + name for name in ("first", "second")]
    for uri, now in ((first, 1), (second, 2), (first, 3), (first, 2)):
        await e.on_post_write(uri + "/messages.jsonl", e.context("a"), now)
    assert e._retry == {("a", "sessions", first): 3}
    assert e.registration_dropped_total == 1
    assert "event=registration_dropped" in caplog.text and "dropped_total=1" in caplog.text
    e._registration_wake.clear()
    await e.on_post_write(first + "/messages.jsonl", e.context("a"), 4)
    assert not e._registration_wake.is_set()  # Updates do not defeat Redis failure backoff.


@pytest.mark.asyncio
async def test_async_worker_drains_bounded_batches(engine, monkeypatch):
    e = engine
    e.batch_size = 512  # Incremental flush still caps the whole round at 256.
    for i in range(257):
        await e.on_post_write(f"viking://user/u/sessions/{i}/messages.jsonl", e.context("a"), i)
    original = e.redis.zadd
    jitter = Mock(return_value=0)
    monkeypatch.setattr("openviking.ttl.runtime.uniform", jitter)
    attempts = 0

    async def register(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("registration temporarily unavailable")
        return await original(*args, **kwargs)

    e.redis.zadd = AsyncMock(side_effect=register)
    await e.start()
    try:
        async with asyncio.timeout(2):
            while e._retry:
                await asyncio.sleep(0.01)
        assert [len(call.args[1]) for call in e.redis.zadd.call_args_list] == [256, 256, 1]
        jitter.assert_called_once_with(0.5, 1)
        assert all(call.kwargs == {"gt": True} for call in e.redis.zadd.call_args_list)
    finally:
        await e.aclose()


@pytest.mark.asyncio
async def test_shutdown_drains_registration_before_redis_close(engine):
    e = engine
    uri = "viking://user/u/sessions/s"
    await e.on_post_write(uri + "/messages.jsonl", e.context("a"), 1)
    redis = e.redis
    original = redis.zadd

    async def register(*args, **kwargs):
        return await original(*args, **kwargs)

    redis.zadd = AsyncMock(side_effect=register)
    redis.aclose = AsyncMock(wraps=redis.aclose)
    await e.aclose()
    redis.zadd.assert_awaited_once_with(e.prefix("a") + "sessions", {uri: 1}, gt=True)
    redis.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_shutdown_registration_failure_is_retained_and_logged(engine, caplog):
    e = engine
    uri = "viking://user/u/sessions/s"
    e.redis.zadd = AsyncMock(side_effect=ConnectionError("redis down"))
    await e.on_post_write(uri + "/messages.jsonl", e.context("a"), 2)
    await e.aclose()
    assert e._retry == {("a", "sessions", uri): 2}
    assert "shutdown left volatile registrations pending=1" in caplog.text


@pytest.mark.asyncio
async def test_backfill_does_not_block_incremental_registration(engine):
    e = engine
    e.batch_size = 1
    root = "viking://user/u/sessions"
    event = "viking://user/u/memories/events/2020/01/01"
    e.fs.ls.return_value = [{"uri": root + "/" + name, "isDir": True} for name in ("a", "b")]

    async def stat(uri, **kwargs):
        if uri == root + "/a/messages.jsonl":
            await e.on_post_write(event + "/file", e.context("a"), 1)
        else:
            async with asyncio.timeout(2):
                while await e.redis.zscore(e.prefix("a") + "user_events", event) is None:
                    await asyncio.sleep(0)
        return {"modTime": "2020-01-01T00:00:00Z"}

    e.fs.stat.side_effect = stat
    await e.register_initial_backfill(TTLBackfillRequest("r", "a", (root,)))
    await e.start()
    assert e._task is None  # Pending backfill alone cannot bypass the hourly callback.
    e._recycle.add("a")
    e._wake_worker()
    try:
        async with asyncio.timeout(2):
            await e._task
        assert not e._retry and e.fs.stat.await_count == 2
    finally:
        await e.aclose()


@pytest.mark.asyncio
async def test_slow_delete_does_not_block_incremental_registration(engine):
    e = engine
    e.retry_limit = 16
    entered, release = asyncio.Event(), asyncio.Event()

    async def slow_rm(*args, **kwargs):
        entered.set()
        await release.wait()

    e.fs.rm.side_effect = slow_rm
    prefix = e.prefix("a")
    await e.redis.hset(
        prefix + "backfill:initial", mapping={"roots": "[]", "checkpoint": '{"root":0,"stack":[]}'}
    )
    await e.redis.set(prefix + "backfill:initial:done", "1")
    await e.redis.zadd(prefix + "user_events", {"viking://user/u/memories/events/2020/01/01": 1})
    await e.start()
    e._recycle.add("a")
    e._wake_worker()
    try:
        async with asyncio.timeout(2):
            await entered.wait()
            for i in range(64):
                uri = f"viking://user/u/sessions/s{i}"
                await e.on_post_write(uri + "/messages.jsonl", e.context("a"), i)
                while await e.redis.zscore(prefix + "sessions", uri) is None:
                    await asyncio.sleep(0)
        assert await e.redis.zcard(prefix + "sessions") == 64
        assert e.registration_dropped_total == 0
        assert not release.is_set() and not e._task.done()
    finally:
        release.set()
        await e.aclose()


@pytest.mark.asyncio
async def test_initial_sorted_batches_resume_after_missing_cursor(engine):
    import json

    e = engine
    e.batch_size = 2
    root = "viking://user/u/sessions"
    uris = [root + "/" + name for name in ("a", "c", "d", "e", "f")]
    e.fs.ls.return_value = [{"uri": u, "isDir": True} for u in reversed(uris)]
    e.fs.stat.return_value = {"modTime": "2020-01-01T00:00:00Z"}
    await e.register_initial_backfill(TTLBackfillRequest("r", "a", (root,)))
    prefix = e.prefix("a") + "backfill:"
    await e.redis.hset(
        prefix + "initial",
        "checkpoint",
        json.dumps({"root": 0, "stack": [{"uri": root, "last": root + "/b"}]}),
    )
    await e._scan("a")
    assert await e.redis.zrange(e.prefix("a") + "sessions", 0, -1) == uris[1:]
    assert await e.redis.get(prefix + "initial:done") == "1"
    e.fs.ls.assert_awaited_once()
    assert e.fs.stat.await_count == 4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "root",
    [
        "viking://user/u/memories/events",
        "viking://user/other/peers/special/memories/events",
    ],
)
async def test_initial_event_parent_listed_once(engine, root, monkeypatch):
    from collections import Counter

    e = engine
    e.batch_interval = 0.01
    sleeps = []
    sleep = asyncio.sleep

    async def paced(delay):
        if delay:
            sleeps.append(delay)
        await sleep(0)

    monkeypatch.setattr("openviking.ttl.runtime.asyncio.sleep", paced)
    e.pair = (TTLConfig(user_events=7, peer_events=7), None)
    tree = {root: [root + "/2020"]}
    year = root + "/2020"
    tree[year] = [year + f"/{m:02d}" for m in range(1, 13)]
    for month in tree[year]:
        tree[month] = [month + "/01"]
    e.fs.ls.side_effect = lambda uri, **kw: [{"uri": child, "isDir": True} for child in tree[uri]]
    await e.register_initial_backfill(TTLBackfillRequest("r", "a", (root,)))
    await e._scan("a")
    assert Counter(c.args[0] for c in e.fs.ls.call_args_list) == Counter(dict.fromkeys(tree, 1))
    kind = "peer_events" if "/peers/" in root else "user_events"
    assert await e.redis.zcard(e.prefix("a") + kind) == 12
    assert await e.redis.get(e.prefix("a") + "backfill:initial:done") == "1"
    assert sleeps == [e.batch_interval] * (len(tree) + 12)
    # An explicit repair still paces Events that are already in the index.
    sleeps.clear()
    await e.register_initial_backfill(TTLBackfillRequest("r", "a", (root,), reset=True))
    await e._scan("a")
    assert sleeps == [e.batch_interval] * (len(tree) + 12)
    e.fs.stat.assert_not_called()


@pytest.mark.asyncio
async def test_user_policy_cache_coalesces_refresh_and_preserves_new_save(engine, monkeypatch):
    from openviking.server.config import UserConfig
    from openviking.server.identity import RequestContext, Role
    from openviking_cli.session.user_id import UserIdentifier

    e = engine
    entered, release = asyncio.Event(), asyncio.Event()

    async def read(*args):
        entered.set()
        await release.wait()
        return UserConfig(ttl=TTLConfig(sessions=1))

    reader = AsyncMock(side_effect=read)
    monkeypatch.setattr("openviking.server.user_config.read_user_config", reader)
    reads = [asyncio.create_task(e.user_policy("a", "u")) for _ in range(8)]
    await entered.wait()
    await e.publish_user_configuration(
        RequestContext(user=UserIdentifier("a", "u"), role=Role.ROOT),
        TTLConfig(sessions=7),
    )
    release.set()
    assert all(c.sessions == 7 for c in await asyncio.gather(*reads))
    reader.assert_awaited_once()
    for _ in range(100):
        assert (await e.user_policy("a", "u")).sessions == 7
    reader.assert_awaited_once()
    e._user_policies[("a", "u")] = (0, TTLConfig(sessions=7))
    assert (await e.user_policy("a", "u")).sessions == 1
    assert reader.await_count == 2
    assert (await e.user_policy("b", "u")).sessions == 1
    assert reader.await_count == 3  # Account is part of the cache key.


@pytest.mark.asyncio
async def test_user_only_ttl_reset_and_restart_reuse_saved_roots(engine, scheduler):
    from openviking.server.identity import RequestContext, Role
    from openviking_cli.session.user_id import UserIdentifier

    e = engine
    e.pair = (None, None)
    ctx = RequestContext(user=UserIdentifier("a", "u"), role=Role.ROOT)
    config = TTLConfig(sessions=7, peer_events=2)
    await e.prepare_user_configuration(ctx, None, config)
    state = await e.read_initial("a")
    assert state["reset_required"] and scheduler.report.await_count == 1
    e._cache_user_policy(("a", "u"), config)
    tree = {
        "viking://user": ["viking://user/u"],
        "viking://user/u/peers": ["viking://user/u/peers/one", "viking://user/u/peers/two"],
    }
    e.fs.ls.side_effect = lambda uri, **kw: [{"uri": u, "isDir": True} for u in tree[uri]]
    await e.register_initial_backfill(TTLBackfillRequest("r", "a"))
    state = await e.read_initial("a")
    assert state["discover"] and state["reset_required"]
    e.fs.ls.assert_not_called()
    e._user_policies.clear()  # Restart before scope discovery must retain the User-only intent.
    e.fs.read_file = AsyncMock(
        return_value=json.dumps({"ttl": config.model_dump(exclude_none=True)})
    )
    await e._resume_initial("a")
    state = await e.read_initial("a")
    assert json.loads(state["roots"]) == [
        "viking://user/u/peers/one/memories/events",
        "viking://user/u/peers/two/memories/events",
        "viking://user/u/sessions",
    ]
    await e.redis.set(e.prefix("a") + "backfill:initial:done", "1")
    e._user_policies.clear()
    e.fs.ls.reset_mock()
    e.fs.read_file = AsyncMock(
        return_value=json.dumps({"ttl": config.model_dump(exclude_none=True)})
    )
    await e.register_initial_backfill(TTLBackfillRequest("r", "a"))
    e.fs.ls.assert_not_called()
    e.fs.stat.assert_not_called()
    e.fs.read_file.assert_awaited_once()
    assert e.fs.read_file.call_args.args[0].endswith("/u/settings/user_config.json")


@pytest.mark.asyncio
@pytest.mark.parametrize("account_days", [30, "disabled"])
async def test_later_user_short_ttl_survives_cache_loss(engine, scheduler, account_days):
    from openviking.ttl.runtime import DAY_MS

    e = engine
    e.pair = (TTLConfig(sessions=30), None)
    prefix = e.prefix("a")
    await e.redis.hset(
        prefix + "backfill:initial", mapping={"roots": '["viking://user/old/sessions"]'}
    )
    await e.redis.set(prefix + "backfill:initial:done", "1")
    ctx = e.context("a")
    config = TTLConfig(sessions=1)
    await e.prepare_user_configuration(ctx, None, config)
    await e.publish_user_configuration(ctx, config)
    uri = "viking://user/ttl-internal/sessions/new"
    written = int(datetime.now(timezone.utc).timestamp() * 1000) - 2 * DAY_MS
    await e.on_post_write(uri + "/messages.jsonl", ctx, written)
    await e._retry_writes()
    e._user_policies.clear()  # Restart/eviction must not change candidate coverage.
    e.pair = (TTLConfig(sessions=account_days), None)
    e.fs.read_file = AsyncMock(
        side_effect=lambda path, **kw: (
            '{"ttl":{"sessions":1}}' if "/ttl-internal/" in path else "{}"
        )
    )
    e.fs.stat.return_value = {
        "modTime": datetime.fromtimestamp(written / 1000, timezone.utc).isoformat()
    }
    e.account_manager.list_users_page.side_effect = [
        {"users": [{"user_id": "old"}]},
        {"users": [{"user_id": "old"}, {"user_id": "ttl-internal"}]},
    ]
    # A user added after this round's snapshot is picked up in the next round.
    assert (await e.recycle_due(TTLRecycleRequest("r", "a"))).deleted == 0
    e.fs.stat.assert_not_called()
    result = await e.recycle_due(TTLRecycleRequest("r", "a"))
    assert result.deleted == 1 and await e.redis.zscore(prefix + "sessions", uri) is None
    assert e.fs.read_file.await_count == 2
    e.account_manager.list_users_page.assert_awaited_with("a", limit=None, expose_key=False)
    e.fs.ls.assert_not_called()


@pytest.mark.asyncio
async def test_hourly_callback_skips_config_and_maintenance_loads_users_once(engine, scheduler):
    from openviking.scheduling.expiry.api import CallbackRequest

    e = engine
    prefix = e.prefix("a") + "backfill:"
    roots = [f"viking://user/u{i}/sessions" for i in range(1025)]
    await e.redis.hset(prefix + "initial", "roots", json.dumps(roots))
    await e.redis.set(prefix + "initial:done", "1")
    await e.redis.zadd(e.prefix("a") + "sessions", {roots[0] + "/new": 2**53 - 1})
    e.account_manager.list_users_page.return_value = {
        "users": [{"user_id": f"u{i}"} for i in range(1025)]
    }
    e.fs.read_file = AsyncMock(return_value="{}")
    for _ in range(2):
        before = e.fs.read_file.await_count
        await e.handle(CallbackRequest(e._identity("a")))
        assert e.fs.read_file.await_count == before
        await e._run()
        assert e.fs.read_file.await_count - before == 1025
    e.fs.ls.assert_not_called()
    e.fs.stat.assert_not_called()


@pytest.mark.asyncio
async def test_maintenance_failure_isolated_to_one_account(engine, scheduler):
    e = engine
    e._recycle.update(("bad", "good"))
    bad, good = list(e._recycle)
    for account in e._recycle:
        await e.redis.set(e.prefix(account) + "backfill:initial:done", "1")
    read = e.read_initial
    failing = True

    async def state(account):
        if account == bad and failing:
            raise ConnectionError("account read failed")
        return await read(account)

    e.read_initial = AsyncMock(side_effect=state)
    await e.start()
    e._wake_worker()
    try:
        async with asyncio.timeout(2):
            while good in e._recycle:
                await asyncio.sleep(0)
        assert e._recycle == {bad}
        failing = False
        e._wake_worker()
        async with asyncio.timeout(2):
            await e._task
        assert not e._recycle
        assert [c.args[0] for c in e.read_initial.await_args_list].count(good) == 1
    finally:
        await e.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target",
    [
        "viking://user/u/sessions/s/history/archive_001",
        "viking://user/u/memories/events/2026/10/28/event.md",
    ],
)
async def test_target_checks_do_not_read_metadata(engine, target):
    e = engine
    e._cache_user_policy(("a", "u"), None)
    e.fs.read_file = AsyncMock(side_effect=AssertionError("TTL target check read metadata"))
    await e.check_target(target, e.context("a"))
    e.fs.stat.assert_awaited_once()
    e.fs.read_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_account_retirement_waits_for_registration_and_allows_recreation(engine, scheduler):
    from openviking.scheduling.expiry.api import CallbackRequest

    e = engine
    uri = "viking://user/u/sessions/s"
    entered, release = asyncio.Event(), asyncio.Event()
    zadd = e.redis.zadd

    async def register(*args, **kwargs):
        entered.set()
        await release.wait()
        return await zadd(*args, **kwargs)

    scheduler.unregister = AsyncMock()
    await e.on_post_write(uri + "/messages.jsonl", e.context("a"), 1)
    e.redis.zadd = register
    flushing = asyncio.create_task(e._retry_writes())
    await entered.wait()
    e.account_manager.get_account.return_value = None
    retiring = asyncio.create_task(e.delete_account("a"))
    await asyncio.sleep(0)
    assert not retiring.done()
    release.set()
    await asyncio.gather(flushing, retiring)
    assert not [key async for key in e.redis.scan_iter(match=e.prefix("a") + "*")]
    assert not e._retry and "a" not in e._scheduled

    e.account_manager.get_account.return_value = {"status": "active"}
    await e.on_post_write(uri + "/messages.jsonl", e.context("a"), 2)
    await e._retry_writes()
    assert await e.redis.zscore(e.prefix("a") + "sessions", uri) == 2
    unregisters = scheduler.unregister.await_count
    e.account_manager.get_account.side_effect = ConnectionError("identity store unavailable")
    with pytest.raises(ConnectionError):
        await e.handle(CallbackRequest(e._identity("a")))
    assert await e.redis.zscore(e.prefix("a") + "sessions", uri) == 2
    assert scheduler.unregister.await_count == unregisters
