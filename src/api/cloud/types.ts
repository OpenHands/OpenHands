/**
 * Minimal slice of the OpenHands cloud organization shape that the GUI needs
 * to render the backend selector. Full shape lives in the OpenHands repo;
 * we keep only the fields actually read by this codebase.
 */
export interface CloudOrganization {
  id: string;
  name: string;
  is_personal?: boolean;
}

export interface CloudOrganizationsResponse {
  items: CloudOrganization[];
  current_org_id: string | null;
}

/**
 * Subset of OpenHands' OrganizationMember that the GUI reads. Returned by
 * `GET /api/organizations/{orgId}/me`. The relationship between the org and
 * the current user is the source of truth for "is this a personal
 * workspace?" — namely `org_id === user_id`.
 */
export interface CloudOrganizationMember {
  org_id: string;
  user_id: string;
  email?: string | null;
  role?: string;
  status?: string;
}

/**
 * Response from `GET /api/keys/current`. Identifies the org the calling
 * API key is scoped to. Legacy keys without an org binding cause the
 * upstream to return HTTP 400 instead of producing this shape — the
 * caller catches that case rather than expecting a partial response.
 */
export interface CloudApiKeyMetadata {
  id: string;
  name: string;
  org_id: string | null;
  user_id: string;
  auth_type: string;
}

/**
 * Response from `GET /api/v1/users/git-info`. Identifies the currently
 * authenticated user across the connected git providers.
 */
export interface CloudGitUser {
  id: string;
  login: string;
  avatar_url: string;
  company: string | null;
  name: string | null;
  email: string | null;
}
