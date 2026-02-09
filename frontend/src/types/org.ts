export type OrganizationUserRole = "member" | "admin" | "owner";

export interface Organization {
  id: string;
  name: string;
  contact_name: string;
  contact_email: string;
  conversation_expiration: number;
  agent: string;
  default_max_iterations: number;
  security_analyzer: string;
  confirmation_mode: boolean;
  default_llm_model: string;
  default_llm_api_key_for_byor: string;
  default_llm_base_url: string;
  remote_runtime_resource_factor: number;
  enable_default_condenser: boolean;
  billing_margin: number;
  enable_proactive_conversation_starters: boolean;
  sandbox_base_container_image: string;
  sandbox_runtime_container_image: string;
  org_version: number;
  mcp_config: {
    tools: unknown[];
    settings: Record<string, unknown>;
  };
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
  llm_api_key: string;
  max_iterations: number;
  llm_model: string;
  llm_api_key_for_byor: string | null;
  llm_base_url: string;
  status: "active" | "invited" | "inactive";
}

/** org_id and user_id are provided via URL params */
export type UpdateOrganizationMemberParams = Partial<
  Omit<OrganizationMember, "org_id" | "user_id">
>;
