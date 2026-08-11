export interface GitSyncStatus {
  enabled: boolean;
  repo_url: string;
  branch: string;
  path: string;
  encryption_enabled: boolean;
  /** Seconds between automatic syncs; 0 means manual-only. */
  interval_seconds: number;
  last_synced_commit: string | null;
  last_synced_at: string | null;
  last_error: string | null;
  last_error_at: string | null;
  dirty_count: number;
}

export interface GitSyncConfigUpdateRequest {
  enabled?: boolean | null;
  interval_seconds?: number | null;
  repo_url?: string | null;
  branch?: string | null;
  path?: string | null;
  token?: string | null;
  encryption_key?: string | null;
  author_name?: string | null;
  author_email?: string | null;
}

export interface GitSyncTriggerResponse {
  triggered: boolean;
}
