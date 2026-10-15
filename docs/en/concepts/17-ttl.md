# Staged TTL delivery

The first stage delivers the internal strict deletion primitive needed by TTL. Policy configuration, expiry checks, read filtering, scanning, failure records and retry scheduling follow separately.

## Internal deletion contract

TTL calls `VikingFS.rm(uri, recursive=True, ctx=ctx, strict=True)` with an internal ROOT context. Other roles cannot use strict mode. HTTP, MCP, SDK and CLI do not expose `strict`; ordinary deletion keeps its existing behavior.

Strict deletion reuses access checks and path locks, then:

1. Deletes vectors within the current account and URI scope, including orphan vectors whose files are already missing. Recursive scope includes the target and descendants, excluding similarly prefixed siblings.
2. Confirms zero vectors using the same filter. An unavailable vector backend, delete/count error or residue fails the operation before files are removed.
3. Removes files, confirms absence with a cache-bypassing stat, and checks vectors again. Only typed absence errors prove completion; a network error containing "not found" does not.

Backend rate limits, outages and timeouts retain their original exceptions and error codes. Residue raises `StorageException(action="confirm_delete", retryable=True)`. Callers must not mark an attempt completed after an exception. `estimated_deleted_count` remains an estimate, not completion evidence.

Repeated calls can finish cleanup by the same URI even after files disappear. ROOT cleanup deletes the vector scope without traversing files to collect vector URIs.

## Later TTL integration

The scheduler will persist account, URI and failure details, bound consecutive retries, and reduce retry frequency for long failures. This primitive adds neither internal retries nor durable task/failure storage. A failed final confirmation is still a failed attempt even if the data was removed; retrying the URI is safe.

This stage confirms the current storage operation. TTL integration must still coordinate expiry rechecks, asynchronous writes and URI reuse so stale work cannot recreate data or delete a replacement object. Shipping this stage does not enable TTL or prove that cloud billing has stopped.
