export type OrganizationUserRole = "user" | "admin" | "owner";

export interface Organization {
  id: string;
  name: string;
  balance: number;
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
