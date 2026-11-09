export type V1SandboxStatus =
  "STARTING" | "RUNNING" | "PAUSED" | "ERROR" | "MISSING";

export interface V1ExposedUrl {
  name: string;
  url: string;
}

export interface V1SandboxInfo {
  id: string;
  created_by_user_id: string | null;
  sandbox_spec_id: string;
  status: V1SandboxStatus;
  session_api_key: string | null;
  exposed_urls: V1ExposedUrl[] | null;
  created_at: string;
  /** Times the sandbox container has restarted; null when unknown. */
  restart_count?: number | null;
  /** Why the sandbox container last terminated, e.g. "OOMKilled". */
  restart_reasons?: string[];
}
