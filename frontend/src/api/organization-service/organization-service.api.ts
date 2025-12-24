import {
  Organization,
  OrganizationMember,
  UpdateOrganizationMemberParams,
} from "#/types/org";
import { openHands } from "../open-hands-axios";

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
    const { data } = await openHands.get<Organization[]>("/api/organizations");
    return data;
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

  getOrganizationMembers: async ({ orgId }: { orgId: string }) => {
    const { data } = await openHands.get<OrganizationMember[]>(
      `/api/organizations/${orgId}/members`,
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
};
