/**
 * Workspace creation types for code vs pentest flows (PROJETOSIN-183).
 * @spec PROJETOSIN-183 — Workspace type selector
 */

export type WorkspaceType = "code" | "pentest";

export type AutonomyMode = "manual" | "semi_autonomous" | "autonomous";

export type PentestRuntimeProfile = "web" | "network" | "mobile" | "sast";

export interface WorkspaceCreationParams {
  type: WorkspaceType;
  name: string;
  workingDir?: string;
  /** Required when type === "pentest". */
  engagementId?: string;
  /** Only meaningful for pentest workspaces. */
  autonomyMode?: AutonomyMode;
}

export interface PentestConversationMetadata {
  workspace_type: "pentest";
  engagement_id: string;
  autonomy_mode: AutonomyMode;
  runtime_profile: PentestRuntimeProfile;
}

export interface PentestEngagementSummary {
  id: string;
  name: string;
  /** ISO timestamp when scope was authorized; null = cannot provision. */
  scope_authorized_at: string | null;
}
