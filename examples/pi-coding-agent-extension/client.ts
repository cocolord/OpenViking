import type { OVConfig } from "./config.js";
import type { OvHttpRequestOptions } from "./shared/ov-http.mjs";
import { createOvHttp } from "./shared/ov-http.mjs";

// --- OV API Response Shapes ---
// All OV responses wrap in: { status: "ok"|"error", result: T, error?: {...}, ... }
// This client normalizes to { ok, result } internally.
//
// Scope: session plumbing only — health, the OV session and its commit, plus the
// raw `fetchJSON` the shared recall/sync/profile modules are built on. Search,
// general content reads, filesystem operations and resource ingest are the
// model's business and reach the server over MCP (`lib/mcp-bridge.mjs`). The
// archive reads below are takeover plumbing: they pin the overview to the
// exact archive returned by commit, so an older session overview cannot move
// the local history boundary.

export interface OVSessionMeta {
  session_id: string;
  message_count: number;
  total_message_count?: number;
  commit_count: number;
  pending_tokens?: number;
  memories_extracted?: Record<string, number>;
  last_commit_at?: string;
}

export interface OVSessionContext {
  latest_archive_overview: string | null;
  pre_archive_abstracts: any[];
  messages: any[];
  estimatedTokens: number;
  stats: {
    totalArchives: number;
    includedArchives: number;
    droppedArchives: number;
    failedArchives: number;
    activeTokens: number;
    archiveTokens: number;
  };
}

export interface OVCommitResult {
  task_id?: string;
  archive_uri?: string;
  trace_id?: string;
}

export interface OVCommitResponse {
  result: OVCommitResult | null;
  traceId?: string;
  error?: any;
  status?: number;
}

export interface OVResponse<T> {
  ok: boolean;
  result: T | null;
  error?: any;
  status?: number;
  traceId?: string;
}

export class OVClient {
  private http: ReturnType<typeof createOvHttp>;
  connected: boolean = false;

  /** Read-only access to config (for value access across modules). */
  readonly cfg: OVConfig;

  constructor(config: OVConfig) {
    this.cfg = config;
    this.http = createOvHttp(
      { ...config, baseUrl: config.endpoint.replace(/\/+$/, "") },
      { defaultTimeoutMs: 10000, resolveActorPeerId: () => config.peerId },
    );
  }

  /** Core fetch wrapper. Returns { ok, result } after parsing OV's { status, result } envelope. */
  async fetchJSON<T>(path: string, init?: RequestInit, options?: OvHttpRequestOptions): Promise<OVResponse<T>> {
    return this.http(path, init, options);
  }

  // ========== Health ==========

  async health(): Promise<boolean> {
    const res = await this.fetchJSON<any>("/health", undefined, { timeoutMs: 5000 });
    this.connected = res.ok;
    return res.ok;
  }

  // ========== Sessions ==========

  /** GET /api/v1/sessions/{id} — session metadata */
  async getSession(sessionId: string, autoCreate = false): Promise<OVSessionMeta | null> {
    const q = autoCreate ? "?auto_create=true" : "";
    const res = await this.fetchJSON<OVSessionMeta>(
      `/api/v1/sessions/${encodeURIComponent(sessionId)}${q}`,
      undefined, { timeoutMs: 5000 },
    );
    return res.ok ? res.result : null;
  }

  /** GET /api/v1/sessions/{id}/context — assembled context with archive overview */
  async getSessionContext(sessionId: string, tokenBudget = 128000): Promise<OVSessionContext | null> {
    const res = await this.fetchJSON<OVSessionContext>(
      `/api/v1/sessions/${encodeURIComponent(sessionId)}/context?token_budget=${tokenBudget}`,
      undefined, { timeoutMs: 10000 },
    );
    return res.ok ? res.result : null;
  }

  /** Read the Working Memory only after this exact archive is fully complete. */
  async readArchiveOverviewResponse(archiveUri: string): Promise<OVResponse<string>> {
    const base = String(archiveUri ?? "").trim().replace(/\/+$/, "");
    if (!base) {
      return {
        ok: false,
        result: null,
        status: 400,
        error: { code: "INVALID_ARCHIVE_URI", message: "archive URI is required" },
      };
    }

    const done = await this.fetchJSON<string>(
      `/api/v1/content/read?uri=${encodeURIComponent(`${base}/.done`)}`,
      undefined, { timeoutMs: 10000 },
    );
    if (!done.ok) {
      if (done.status !== 404) return done;
      return {
        ...done,
        error: {
          code: "ARCHIVE_NOT_READY",
          message: "archive completion marker is not available",
          cause: done.error,
        },
      };
    }
    if (typeof done.result !== "string" || !done.result.trim()) {
      return {
        ...done,
        ok: false,
        result: null,
        error: { code: "ARCHIVE_NOT_READY", message: "archive completion marker is empty" },
      };
    }
    try {
      const marker = JSON.parse(done.result);
      if (marker?.working_memory_enabled === false) {
        return {
          ...done,
          ok: false,
          result: null,
          error: { code: "ARCHIVE_OVERVIEW_DISABLED", message: "working memory is disabled for this archive" },
        };
      }
    } catch {
      // Legacy markers may not be JSON; their overview remains authoritative.
    }

    const overview = await this.fetchJSON<string>(
      `/api/v1/content/overview?uri=${encodeURIComponent(base)}`,
      undefined, { timeoutMs: 10000 },
    );
    if (!overview.ok) return overview;
    const body = typeof overview.result === "string" ? overview.result.trim() : "";
    if (!body || /\[Directory overview is not ready\]$/.test(body)) {
      return {
        ...overview,
        ok: false,
        result: null,
        error: { code: "ARCHIVE_OVERVIEW_NOT_READY", message: "archive overview is not ready" },
      };
    }
    return { ...overview, result: body };
  }

  /** POST /api/v1/sessions/{id}/commit — commit session for archiving + extraction */
  async commitSessionResponse(
    sessionId: string,
    keepRecentCount = this.cfg.commitKeepRecentCount,
  ): Promise<OVCommitResponse> {
    const res = await this.fetchJSON<OVCommitResult>(
      `/api/v1/sessions/${encodeURIComponent(sessionId)}/commit`,
      { method: "POST", body: JSON.stringify({ keep_recent_count: keepRecentCount }) },
      { timeoutMs: 30000 },
    );
    if (res.ok && res.result && !res.result.trace_id && res.traceId) {
      res.result.trace_id = res.traceId;
    }
    return {
      result: res.ok ? res.result : null,
      traceId: res.traceId,
      error: res.error,
      status: res.status,
    };
  }
}
