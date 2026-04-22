export type OrganizationUserRole = "member" | "admin" | "owner";

export interface Organization {
  id: string;
  name: string;
  contact_name: string;
  contact_email: string;
  conversation_expiration: number;
  remote_runtime_resource_factor: number;
  billing_margin: number;
  enable_proactive_conversation_starters: boolean;
  sandbox_base_container_image: string;
  sandbox_runtime_container_image: string;
  org_version: number;
  agent_settings?: Record<string, unknown>;
  search_api_key: string | null;
  sandbox_api_key: string | null;
  max_budget_per_task: number;
  enable_solvability_analysis: boolean;
  v1_enabled: boolean;
  credits: number;
  is_personal?: boolean;
}

export interface OrganizationMember {
  org_id: string;
  user_id: string;
  email: string;
  role: OrganizationUserRole;
  max_iterations: number;
  llm_model: string;
  llm_base_url: string;

  llm_api_key: string;
  agent_settings?: Record<string, unknown>;
  status: "active" | "invited" | "inactive";
}

export interface OrganizationMembersPage {
  items: OrganizationMember[];
  current_page: number;
  per_page: number;
}

/** org_id and user_id are provided via URL params */
export type UpdateOrganizationMemberParams = Partial<
  Omit<OrganizationMember, "org_id" | "user_id">
>;

/**
 * Query data structure for the organizations query.
 * This represents the raw data returned by queryClient before any `select` transform.
 */
export type OrganizationsQueryData = {
  items: Organization[];
  currentOrgId: string | null;
};

export interface GitOrgClaim {
  id: string;
  org_id: string;
  provider: string;
  git_organization: string;
  claimed_by: string;
  claimed_at: string;
}

export interface UserGitOrganizationsResponse {
  provider: string;
  organizations: string[];
}

export interface GitOrg {
  id: string;
  claimId: string | null;
  provider: string;
  name: string;
  status: "unclaimed" | "claimed" | "claiming" | "disconnecting";
}
