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

export interface BitbucketDCResource {
  project_key: string;
  repo_slug: string;
  name: string;
  full_name: string;
  type: "repository";
  connection_id: number | null;
  webhook_enrolled: boolean;
  webhook_id: string | null;
  webhook_url: string | null;
  webhook_secret_set: boolean;
  installed_by_user_id: string | null;
  last_synced: string | null;
}

export interface BitbucketDCResourcesResponse {
  resources: BitbucketDCResource[];
}

export interface BitbucketDCResourceIdentifier {
  project_key: string;
  repo_slug: string;
}

export interface EnrollBitbucketDCWebhookRequest {
  resource: BitbucketDCResourceIdentifier;
}

export interface BitbucketDCWebhookEnrollmentResult {
  project_key: string;
  repo_slug: string;
  success: boolean;
  error: string | null;
  connection_id: number | null;
  webhook_url: string | null;
  webhook_secret: string | null;
  webhook_name: string;
  events: string[];
}

export interface UpdateBitbucketDCWebhookIdRequest {
  resource: BitbucketDCResourceIdentifier;
  webhook_id: string;
}

export interface BitbucketDCWebhookIdUpdateResult {
  project_key: string;
  repo_slug: string;
  success: boolean;
  error: string | null;
}

export interface BitbucketDCWebhookRequest {
  resource: BitbucketDCResourceIdentifier;
}

export interface BitbucketDCWebhookInstallationResult {
  project_key: string;
  repo_slug: string;
  success: boolean;
  error: string | null;
  webhook_id: string | null;
  connection_id: number | null;
  webhook_url: string | null;
}
