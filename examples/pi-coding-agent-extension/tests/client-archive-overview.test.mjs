import test from "node:test";
import assert from "node:assert/strict";

import { OVClient } from "../client.ts";

function makeClient() {
  return new OVClient({
    endpoint: "http://127.0.0.1:1933",
    peerId: "pi-test",
    commitKeepRecentCount: 3,
  });
}

test("archive overview waits for exact completion and uses the owned overview endpoint", async () => {
  const client = makeClient();
  const requests = [];
  client.fetchJSON = async (path, init, options) => {
    requests.push({ path, init, options });
    return requests.length === 1
      ? { ok: true, status: 200, result: "{}" }
      : { ok: true, status: 200, result: "fresh overview\n" };
  };

  const base = "viking://user/test/sessions/pi-1/history/archive_002";
  const result = await client.readArchiveOverviewResponse(`${base}/`);

  assert.deepEqual(result, { ok: true, status: 200, result: "fresh overview" });
  assert.deepEqual(requests, [
    {
      path: `/api/v1/content/read?uri=${encodeURIComponent(`${base}/.done`)}`,
      init: undefined,
      options: { timeoutMs: 10000 },
    },
    {
      path: `/api/v1/content/overview?uri=${encodeURIComponent(base)}`,
      init: undefined,
      options: { timeoutMs: 10000 },
    },
  ]);
});

test("archive overview preserves failures and never reads a pending archive overview", async () => {
  const client = makeClient();
  const failure = { ok: false, status: 401, result: null, error: { code: "UNAUTHORIZED" } };
  let calls = 0;
  client.fetchJSON = async () => { calls++; return failure; };

  assert.equal(await client.readArchiveOverviewResponse("viking://archive/2"), failure);
  assert.equal(calls, 1);
});

test("archive overview labels a missing completion marker as pending", async () => {
  const client = makeClient();
  const cause = { code: "NOT_FOUND", message: "file not found" };
  client.fetchJSON = async () => ({ ok: false, status: 404, result: null, error: cause });

  const result = await client.readArchiveOverviewResponse("viking://archive/2");
  assert.equal(result.ok, false);
  assert.equal(result.status, 404);
  assert.equal(result.error.code, "ARCHIVE_NOT_READY");
  assert.equal(result.error.cause, cause);
});

test("archive overview rejects missing or placeholder content after completion", async () => {
  const client = makeClient();
  const responses = [
    { ok: true, status: 200, result: "{}" },
    { ok: true, status: 200, result: "# archive_002\n\n[Directory overview is not ready]" },
  ];
  client.fetchJSON = async () => responses.shift();

  const result = await client.readArchiveOverviewResponse("viking://archive/2");
  assert.equal(result.ok, false);
  assert.equal(result.error.code, "ARCHIVE_OVERVIEW_NOT_READY");
  assert.equal(result.result, null);
});

test("archive overview stops after the completion marker when working memory is disabled", async () => {
  const client = makeClient();
  let calls = 0;
  client.fetchJSON = async () => {
    calls++;
    return { ok: true, status: 200, result: '{"working_memory_enabled":false}' };
  };

  const result = await client.readArchiveOverviewResponse("viking://archive/2");
  assert.equal(result.ok, false);
  assert.equal(result.error.code, "ARCHIVE_OVERVIEW_DISABLED");
  assert.equal(calls, 1);
});
