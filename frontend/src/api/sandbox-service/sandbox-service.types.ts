// sandbox-service.types.ts
// This file contains types for Sandbox API.

export type V1SandboxStatus =
  | "MISSING"
  | "STARTING"
  | "RUNNING"
  | "STOPPED"
  | "PAUSED";

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
}
