import type { OVClient } from "./client.js";
import type { OVConfig } from "./config.js";
import type { SyncManager } from "./sync.js";
import { TakeoverCore } from "./lib/takeover-core.mjs";

export function createTakeoverManager(opts: {
  pi: any;
  client: OVClient;
  sync: SyncManager;
  config: OVConfig;
  log?: (message: string) => void;
}): TakeoverCore {
  const { pi, client, sync, config } = opts;
  return new TakeoverCore({
    config,
    io: {
      flush: () => sync.flushForTakeover(),
      commit: (commitOpts?: { queueOnFailure?: boolean; keepRecentCount?: number }) => sync.commit(commitOpts),
      fetchOverview: async (archiveUri: string) => {
        try {
          const response = await client.readArchiveOverviewResponse(archiveUri);
          if (!response.ok) {
            const normalWait = response.error?.code === "ARCHIVE_NOT_READY";
            if (!normalWait) {
              opts.log?.(
                `takeover: archive overview read failed (${response.status ?? 0} ${response.error?.code || "unknown"})`,
              );
            }
            return normalWait ? "" : null;
          }
          return response.result ?? "";
        } catch (error) {
          opts.log?.(`takeover: archive overview read failed (${error instanceof Error ? error.message : String(error)})`);
          return null;
        }
      },
      persistEntry: (customType: string, data: any) => {
        if (typeof pi?.appendEntry === "function") {
          pi.appendEntry(customType, data);
        }
      },
      getWatermark: () => sync.syncedCount,
      log: opts.log,
    },
  });
}
