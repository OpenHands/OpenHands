export interface GitLabResource {
  id: string;
  name: string;
  full_path: string;
  type: "project" | "group";
  webhook_installed: boolean;
  webhook_uuid: string | null;
  last_synced: string | null;
}

export interface GitLabResourcesResponse {
  resources: GitLabResource[];
}

export interface ResourceIdentifier {
  type: "project" | "group";
  id: string;
}

export interface ReinstallWebhookRequest {
  resource: ResourceIdentifier;
}

export interface ResourceInstallationResult {
  resource_id: string;
  resource_type: string;
  success: boolean;
  error: string | null;
}
