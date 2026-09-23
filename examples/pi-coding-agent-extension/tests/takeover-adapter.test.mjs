import test from "node:test";
import assert from "node:assert/strict";

import { createTakeoverManager } from "../takeover.ts";

test("takeover adapter reads the overview for the archive returned by commit", async () => {
  const calls = [];
  const archiveUri = "viking://user/test/sessions/pi-1/history/archive_002";
  const manager = createTakeoverManager({
    pi: { appendEntry: (type, data) => calls.push({ type, data }) },
    client: {
      readArchiveOverviewResponse: async (uri) => {
        calls.push({ overviewUri: uri });
        return { ok: true, status: 200, result: "fresh overview" };
      },
    },
    sync: {
      flushForTakeover: async () => true,
      commit: async () => ({ task_id: "task-2", archive_uri: archiveUri }),
      syncedCount: 7,
    },
    config: {
      takeoverEnabled: true,
      takeoverTokenThreshold: 1,
      takeoverKeepRecentTurns: 1,
      takeoverOverviewBudget: 1000,
      takeoverOverviewPollMs: 0,
      takeoverOverviewPollMax: 1,
    },
  });

  manager.transformContext([
    { role: "user", content: "one" },
    { role: "assistant", content: "answer" },
    { role: "user", content: "two" },
  ]);

  assert.equal(await manager.onTurnSynced(1), true);
  assert.deepEqual(calls.find((call) => call.overviewUri), { overviewUri: archiveUri });
  assert.equal(manager.state.overview, "fresh overview");
  assert.equal(manager.state.coveredUserTurns, 1);
  assert.equal(manager.state.syncedEntryCount, 7);
});

test("takeover adapter preserves a client failure without advancing", async () => {
  const logs = [];
  let overviewCalls = 0;
  const manager = createTakeoverManager({
    pi: {},
    client: {
      readArchiveOverviewResponse: async () => {
        overviewCalls++;
        return { ok: false, status: 401, result: null, error: { code: "UNAUTHORIZED" } };
      },
    },
    sync: {
      flushForTakeover: async () => true,
      commit: async () => ({ archive_uri: "viking://archive/2" }),
      syncedCount: 0,
    },
    config: {
      takeoverEnabled: true,
      takeoverKeepRecentTurns: 1,
      takeoverOverviewPollMs: 0,
      takeoverOverviewPollMax: 3,
    },
    log: (message) => logs.push(message),
  });
  manager.coveredUserTurns = 1;
  manager.overview = "previous overview";
  manager.lastSeenUserTurns = 4;

  assert.equal(await manager.commitAndAdvance(), false);
  assert.equal(manager.state.coveredUserTurns, 1);
  assert.equal(manager.state.overview, "previous overview");
  assert.equal(overviewCalls, 1);
  assert.ok(logs.some((message) => message.includes("401 UNAUTHORIZED")));
});

test("takeover adapter does not repoll a terminal archive with no usable overview", async () => {
  let overviewCalls = 0;
  const manager = createTakeoverManager({
    pi: {},
    client: {
      readArchiveOverviewResponse: async () => {
        overviewCalls++;
        return {
          ok: false,
          status: 200,
          result: null,
          error: { code: "ARCHIVE_OVERVIEW_NOT_READY" },
        };
      },
    },
    sync: {
      flushForTakeover: async () => true,
      commit: async () => ({ archive_uri: "viking://archive/2" }),
      syncedCount: 0,
    },
    config: {
      takeoverEnabled: true,
      takeoverKeepRecentTurns: 1,
      takeoverOverviewPollMs: 0,
      takeoverOverviewPollMax: 15,
    },
  });
  manager.coveredUserTurns = 1;
  manager.overview = "previous overview";
  manager.lastSeenUserTurns = 4;

  assert.equal(await manager.commitAndAdvance(), false);
  assert.equal(overviewCalls, 1);
  assert.equal(manager.state.coveredUserTurns, 1);
  assert.equal(manager.state.overview, "previous overview");
});

test("takeover adapter keeps polling a missing completion marker", async () => {
  let overviewCalls = 0;
  const manager = createTakeoverManager({
    pi: {},
    client: {
      readArchiveOverviewResponse: async () => {
        overviewCalls++;
        return overviewCalls === 1
          ? { ok: false, status: 404, result: null, error: { code: "ARCHIVE_NOT_READY" } }
          : { ok: true, status: 200, result: "fresh overview" };
      },
    },
    sync: {
      flushForTakeover: async () => true,
      commit: async () => ({ archive_uri: "viking://archive/2" }),
      syncedCount: 0,
    },
    config: {
      takeoverEnabled: true,
      takeoverKeepRecentTurns: 1,
      takeoverOverviewPollMs: 0,
      takeoverOverviewPollMax: 3,
    },
  });
  manager.lastSeenUserTurns = 3;

  assert.equal(await manager.commitAndAdvance(), true);
  assert.equal(overviewCalls, 2);
  assert.equal(manager.state.overview, "fresh overview");
});
