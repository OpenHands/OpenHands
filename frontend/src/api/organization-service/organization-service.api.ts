import {
  GitOrgClaim,
  Organization,
  OrganizationMember,
  OrganizationMembersPage,
  UpdateOrganizationMemberParams,
} from "#/types/org";
import { Settings } from "#/types/settings";
import { openHands } from "../open-hands-axios";

type OrganizationAgentSettingsResponse = Pick<
  Settings,
  | "agent_settings"
  | "conversation_settings"
  | "search_api_key"
  | "llm_api_key_set"
>;

export const organizationService = {
  getMe: async ({ orgId }: { orgId: string }) => {
    const { data } = await openHands.get<OrganizationMember>(
      `/api/organizations/${orgId}/me`,
    );

    return data;
  },

  getOrganization: async ({ orgId }: { orgId: string }) => {
    const { data } = await openHands.get<Organization>(
      `/api/organizations/${orgId}`,
    );
    return data;
  },

  getOrganizations: async () => {
    const { data } = await openHands.get<{
      items: Organization[];
      current_org_id: string | null;
    }>("/api/organizations");
    return {
      items: data?.items || [],
      currentOrgId: data?.current_org_id || null,
    };
  },

  updateOrganization: async ({
    orgId,
    name,
  }: {
    orgId: string;
    name: string;
  }) => {
    const { data } = await openHands.patch<Organization>(
      `/api/organizations/${orgId}`,
      { name },
    );
    return data;
  },

  deleteOrganization: async ({ orgId }: { orgId: string }) => {
    await openHands.delete(`/api/organizations/${orgId}`);
  },

  getOrganizationMembers: async ({
    orgId,
    page = 1,
    limit = 10,
    email,
  }: {
    orgId: string;
    page?: number;
    limit?: number;
    email?: string;
  }) => {
    const params = new URLSearchParams();

    // Calculate offset from page number (page_id is offset-based)
    const offset = (page - 1) * limit;
    params.set("page_id", String(offset));
    params.set("limit", String(limit));

    if (email) {
      params.set("email", email);
    }

    const { data } = await openHands.get<OrganizationMembersPage>(
      `/api/organizations/${orgId}/members?${params.toString()}`,
    );

    return data;
  },

  getOrganizationMembersCount: async ({
    orgId,
    email,
  }: {
    orgId: string;
    email?: string;
  }) => {
    const params = new URLSearchParams();

    if (email) {
      params.set("email", email);
    }

    const { data } = await openHands.get<number>(
      `/api/organizations/${orgId}/members/count?${params.toString()}`,
    );

    return data;
  },

  getOrganizationPaymentInfo: async ({ orgId }: { orgId: string }) => {
    const { data } = await openHands.get<{
      cardNumber: string;
    }>(`/api/organizations/${orgId}/payment`);
    return data;
  },

  updateMember: async ({
    orgId,
    userId,
    ...updateData
  }: {
    orgId: string;
    userId: string;
  } & UpdateOrganizationMemberParams) => {
    const { data } = await openHands.patch(
      `/api/organizations/${orgId}/members/${userId}`,
      updateData,
    );

    return data;
  },

  removeMember: async ({
    orgId,
    userId,
  }: {
    orgId: string;
    userId: string;
  }) => {
    await openHands.delete(`/api/organizations/${orgId}/members/${userId}`);
  },

  inviteMembers: async ({
    orgId,
    emails,
  }: {
    orgId: string;
    emails: string[];
  }) => {
    const { data } = await openHands.post<OrganizationMember[]>(
      `/api/organizations/${orgId}/members/invite`,
      {
        emails,
      },
    );

    return data;
  },

  switchOrganization: async ({ orgId }: { orgId: string }) => {
    const { data } = await openHands.post<Organization>(
      `/api/organizations/${orgId}/switch`,
    );
    return data;
  },

  acceptInvitation: async ({ token }: { token: string }) => {
    const { data } = await openHands.post<{
      success: boolean;
      org_id: string;
      org_name: string;
      role: string;
    }>("/api/organizations/members/invite/accept", { token });

    return data;
  },

  getOrganizationAgentSettings: async () => {
    const { data } = await openHands.get<OrganizationAgentSettingsResponse>(
      "/api/organizations/llm",
    );
    return data;
  },

  saveOrganizationAgentSettings: async (
    settings: Partial<Settings> & Record<string, unknown>,
  ) => {
    const { data } = await openHands.post<OrganizationAgentSettingsResponse>(
      "/api/organizations/llm",
      settings,
    );
    return data;
  },

  getGitClaims: async ({ orgId }: { orgId: string }) => {
    const { data } = await openHands.get<GitOrgClaim[]>(
      `/api/organizations/${orgId}/git-claims`,
    );
    return data;
  },

  claimGitOrg: async ({
    orgId,
    provider,
    gitOrganization,
  }: {
    orgId: string;
    provider: string;
    gitOrganization: string;
  }) => {
    const { data } = await openHands.post<GitOrgClaim>(
      `/api/organizations/${orgId}/git-claims`,
      { provider, git_organization: gitOrganization },
    );
    return data;
  },

  disconnectGitOrg: async ({
    orgId,
    claimId,
  }: {
    orgId: string;
    claimId: string;
  }) => {
    await openHands.delete(`/api/organizations/${orgId}/git-claims/${claimId}`);
  },
};
