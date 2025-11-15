export type OrganizationUserRole = "user" | "admin" | "superadmin";

export interface Organization {
  id: string;
  name: string;
  balance: number;
}

export interface OrganizationMember {
  id: string;
  email: string;
  role: OrganizationUserRole;
  status: "active" | "invited";
}
