# Directory TTL

TTL is off by default. It covers user and peer `events/YYYY/MM/DD` directories and `sessions/{session_id}`. Resources and other memory categories are outside this scope.

## Configuration and policy application

TTL uses instance defaults and Account overrides (a Web Studio "library" is an Account). Configure global defaults, type defaults, or overrides for these user roots:

- `viking://user/{user_id}/memories/events`
- `viking://user/{user_id}/sessions`

The user roots above can have individual overrides. All Peer events share the current Account’s effective `peer_events` policy; setting an individual Peer root in `directories` returns `400 INVALID_ARGUMENT`. Sessions belong to users, with no peer sessions root. User roots without overrides inherit their type default, including newly created users; new peers inherit `peer_events`.

For each node, merge Account overrides → instance runtime settings → startup settings. Then select concrete root → type (`user_events`, `peer_events`, `sessions`) → `global`. An Account `global` policy does not override a more specific instance type default: overriding instance `sessions=30 days` requires an Account `sessions` policy. User and peer identities locate roots; they add no configuration layer.

`disabled` stops inheritance at that node; `inherit` skips it. PATCH `null` removes the current layer's override. Unconfigured types inherit `global`, which defaults to disabled. The optional recommended preset is 60 days for User events, 60 days for Peer events, and 30 days for Sessions. It takes effect only when explicitly selected.

Years, months, dates, individual sessions, nested directories and files expose read-only deadlines. Enabling or changing a root policy also updates existing live directories, including previously unmanaged history, while respecting more-specific overrides. Relative expiry uses the original business timestamp plus the new duration; absolute expiry uses the configured deadline. Shortening may expire a directory immediately. Disabling the effective policy clears live deadlines. Expired and deleted objects are never revived.

The configuration request lists each parent's child directories once, keeps their names, and updates at most eight directories concurrently. Deleting an earlier directory cannot shift later ones out of the work list. The request awaits metadata writes; partial failures report incomplete directories so the same configuration can be retried. For relative policies, history without a reliable original timestamp is reported rather than assigned the configuration update time or directory modTime. Explicit empty event directories can be created and start their lifetime on the first body write.

Policy application and new directory initialization read current settings from the configured source. Each account's batch reuses one resolved policy; ordinary content updates use the saved lifetime. These reads leave the runtime configuration cache and its refresh loop unchanged. The directory name list uses memory proportional to the number of immediate children of the current parent.

## Deadline calculation

Each lifecycle directory stores one `expires_at` in `.meta.json`. Relative retention uses a fixed starting time: the existing `created_at` for Sessions, and `received_at` for the first successful event body write. `ttl_days` belongs only to policy configuration. Events can still read legacy `.ttl.json`. AGFS directory metadata updates preserve other business fields in the same file; directory stat exposes its own deadline.

- Events start their lifetime on the first successful content write. The path date only groups events. Later body writes never renew deadlines; explicit root policy changes can adjust live directories.
- Sessions inherit their root policy on creation. Relative deadlines are creation time plus the configured duration; absolute deadlines use the configured timestamp. Appends, completed commits and task replays leave the deadline unchanged.
- Automatic renewal and renewal recovery are deferred. Users can change library or root policies before expiry to update live directories. Relative deadlines retain the fixed starting time rather than using the policy-change time. Active Sessions still expire at their persisted deadline.

Messages, attachments, archives and L0/L1/L2 share the directory deadline. There is no message-level JSONL retention, `ttl_generation`, per-session override or per-file mode.

TTL adds no deadline fields to body/summary formats or extraction Context objects. Reads obtain the deadline from the owner directory; similarly named body fields remain user content. The first Event write registers its deadline before publishing the body and rolls back a failed write. It needs no extra journal or second metadata write after success.

## Visibility

At `now >= expires_at` in UTC, the directory and all descendants become invisible. Direct reads return 404. Session/file listings, find/search/recall, grep and glob filter expired content and refill visible candidates.

Structured objects expose their owner's `expires_at`, explicitly `null` without TTL. Mixed results carry per-item deadlines. Single-owner text/list responses include the deadline in the envelope. URI-only listing compatibility modes and download bytes keep their existing shape and still enforce server filtering. Root/year/month containers have no shared expiry; policy roots also expose `policy` and `effective_policy`.

Snapshot reads use the live directory deadline, or the snapshot deadline after deletion. Raw import and restore reject overwrites of directories that still have TTL, and reject expired source data, so restoration cannot detach content from its deadline. Each affected directory is checked once, avoiding repeated metadata reads for sibling files.

## Cleanup and performance

Cleanup uses the existing Session commit QueueFS worker framework. The scheduler scans owner directory metadata in accounts that have used TTL, sends expired owners to the queue, and continues in bounded batches while the queue drains. After finishing a pass it waits one day by default. The account marker stores no object deadlines; there is no per-object scheduling index or claim lease.

Initialization, policy changes and cleanup share the existing write coordination lock: the date directory’s `.overview.md` for events and the Session root for sessions. Ordinary bodies acquire no separate metadata admission lock; actual metadata persistence still takes its file write lock. The worker rechecks the deadline under the coordination lock. It deletes files under individual exact locks, retries contention, and uses no tree lock or per-file expiry decision. It first deletes and confirms all vectors under the account and owner URI, then removes owned bodies, messages, attachments and L0/L1. It confirms body removal before deleting owner metadata and finally verifies that the directory is gone. Vector records need no `expires_at` field; the existing account and URI fields cover the entire subtree, including vectors whose source file is already missing. External parent summaries stay unchanged. Deletion triggers no LLM, embedding or summary rebuild.

Exhausted lock-contention retries, vector deletion failure or remaining body files preserve the owner deadline for the next pass. Restarting the scheduler rediscovers candidates from directory metadata; it needs no durable scan cursor or task-history lookup. A failure of the final confirmation request still reports an error even if the data was already deleted. Pre-existing vector-only orphans whose owner metadata is also gone cannot be discovered by a directory scan; strict deletion can remove them when their owner URI is known.

Scanning costs O(owner directories in accounts that have used TTL) per pass. Each page examines at most `batch_size` owners, checks a time budget between owners, and waits for queued work to drain. This bounds queued deletion work, while storage listing latency and backlog can extend the daily pass. It avoids maintaining a second expiry record on writes, transfers, and restores.

System strict cleanup deletes vectors directly by directory URI scope, avoiding a separate file-tree traversal to collect vector URIs. File deletion still takes individual locks and confirms the result.

When vector candidates include Events or Sessions, the account TTL marker is checked after the first query. An absent marker means the account has never had managed TTL: results receive `expires_at: null` without directory metadata reads, and deletion consistency relies on the existing vector deletion path. Absence is not cached, so subsequent requests detect TTL enabled by another worker; writers publish the marker before persisting deadlines. Previously managed accounts still check stored deadlines when their current configuration disables TTL. Resources, skills and empty results add no marker reads.

For managed accounts, vector queries share owner metadata reads across candidate refill rounds and check directory expiry without checking each body or summary file. Retrieval and display read metadata directly through AGFS, omitting the preliminary existence stat. If CacheFS is already configured, its content cache and write/delete invalidation are reused; TTL neither enables nor adds a cache. Expiry is still compared against the current time. Writes, policy changes and cleanup retain their existing existence checks. Legacy objects without metadata require an existing owner directory. Refill excludes the entire expired or deleted owner subtree while retaining live candidates from other directories. The first query size is unchanged. When expired candidates dominate and too few live results remain, subsequent batches grow up to 256 records, or retain the original request size if larger. Sparse expiry does not enlarge batches; refill stops once enough live results are available. Individual file deletion relies on the existing vector deletion path. Directory `count` uses the backend total and converges after physical cleanup, avoiding a full vector scan for real-time expiry counts. Returned content still enforces expiry immediately. Billing may lag physical deletion. OV cleanup alone does not verify cloud billing, gateway forwarding or backup erasure.

Default AGFS reads do not provide a cross-request metadata content cache. The S3/TOS plugin caches directory listings and stat results, but not the contents of `.meta.json`; local reads may benefit from the operating system page cache. Vector queries already reuse owner reads within one query. This TTL change neither configures a cache Provider nor adds a process-local expiry cache.

### Bounded reads and contention retries

**Unresolved performance gate:** The native-pagination marker checks described below are a local draft, not an accepted default-off design. Never-managed accounts must preserve the pre-TTL query path without additional storage calls or TTL budgets. The current runtime configuration manager polls every 30 seconds and provides no all-reader activation barrier. Do not cache marker absence until activation and restart semantics are settled; the current draft does not satisfy this gate.

Accounts without a TTL marker retain native offset pagination. For nonzero offsets, check the marker before the query and again afterwards; if TTL was first enabled during the query, discard the page and restart with TTL filtering. Marker absence is never cached. Zero-offset queries retain the existing post-query check.

TTL-filtered reads have internal per-request limits of 8,192 candidate rows and 1,024 excluded owners. Count repeated rows and the initial `offset + limit` request; any speculative native page also consumes the candidate budget if filtering must restart. Before each backend call, reserve its requested rows. If the next call or exclusion filter would exceed the budget before a complete page or source exhaustion is established, raise `RESOURCE_EXHAUSTED` with reason `ttl_query_budget_exceeded`; never silently truncate results. These are initial operational guardrails, not measured latency guarantees, and require backend load validation. Native pages returned without TTL filtering are not subject to these TTL-specific limits.

Cleanup retries only lock contention, at most twice within the same delivery. Release all attempt locks before jittered backoff (up to 0.25 and 0.75 seconds), then reacquire locks and reread the authoritative deadline. Keep one task identity. Persistent contention remains `skipped: busy` and leaves metadata for the next daily pass; other failures retain existing reporting. This improves transient contention without promising hourly deletion or billing completion. Backoff can add at most one second per contended delivery, excluding storage I/O.

### Avoiding repeated reads (2026-10-09)

These paths reuse data and scope decisions within a request. They add no cache Provider, vector field or lock, and do not cache deadlines across requests:

- Local `glob` fallback shares one `TTLView` across visibility checks, detail stat calls and response annotation, reading each owner's deadline once.
- Native Session `grep` shares one `TTLView` across its initial stat, result filtering and the service's `expires_at` annotation. Other grep fallback paths are outside this optimization.
- `Session.update_config` reads metadata once under the existing Session root lock, then rejects expiry against the current time. Missing-file handling, legacy compatibility and write locks are retained.
- Policy application reuses existing precedence rules to decide whether peer_events is affected before listing peers. Sessions-only changes skip that listing; global inheritance, peer type changes, removed overrides and identical-patch retries retain their existing behavior.

### Discussion only: vector expiry field (on hold, 2026-10-09)

The candidate design copies the owner directory's `expires_at` into vector records and adds a scalar index, allowing expiry filtering during retrieval to reduce metadata reads and candidate refill. Directory metadata remains authoritative. Physical cleanup still uses account and URI scope without requiring this field. This proposal is recorded for discussion; implementation and migration are on hold.

Adding the field and query predicate is relatively small work. Supporting existing data and policy changes has medium-to-high complexity:

- Existing records need backfilling; new writes, reindexing, copies and restores must maintain the field. Old schemas or incomplete backfills cannot safely switch to prefiltering. Missing values must be distinguished from disabled TTL.
- Policy changes must update all vectors belonging to affected directories. When TTL is extended or disabled, an old vector deadline can exclude valid results that a later metadata check cannot recover. Shortening TTL can admit expired results. Visibility delay, partial failures and concurrent writes need an explicit contract.
- Scalar updates need no new embeddings, but still write data and indexes. Added policy-update work grows with the number of associated vectors. The existing local update path merges old records and updates indexes.

Local backends already support field and scalar-index upgrades. Dedicated enterprise cloud collections can use management APIs; personal cloud collections share a physical schema that requires platform coordination. Before resuming, establish ownership of the shared-schema upgrade, existing data volume and policy visibility requirements, then measure cloud update throughput and retrieval gains. Automatic renewal, a new cache Provider and a scheduling index remain outside this change.

## Interfaces

- [TTL configuration](../configuration/01-server.md#ttl): library/type/root policies.
- [Expiry query](../api/12-content.md#document-expiry): `GET /api/v1/fs/stat`, SDK `stat`, CLI `ov stat`.
- [Sessions](../api/05-sessions.md#session-ttl): create/config APIs inherit the root policy and accept no TTL input.

Asynchronous Session commit writes validate the original Phase 1 `task_id` under a short Session lock. Old work cannot modify a replacement Session with the same ID or recreate events from a deleted source. A fresh Session may still import an older calendar date. Direct writes to existing events release the coordination lock after admission; the first write retains it through success or rollback. Ordinary commits reuse their outer write lease. Session appends and commit writes reuse metadata already read under the current root lock, without reacquiring that lock or caching across requests. Admission still checks expiry against the current time. Cleanup blocks new admissions and checks existing file leases, including writes whose file does not yet exist.
