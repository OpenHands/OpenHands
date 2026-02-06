import { http, HttpResponse } from "msw";
import {
  Organization,
  OrganizationMember,
  OrganizationUserRole,
  UpdateOrganizationMemberParams,
} from "#/types/org";

const MOCK_ME: Omit<OrganizationMember, "role" | "org_id"> = {
  user_id: "99",
  email: "me@acme.org",
  llm_api_key: "**********",
  max_iterations: 20,
  llm_model: "gpt-4",
  llm_api_key_for_byor: null,
  llm_base_url: "https://api.openai.com",
  status: "active",
};

export const createMockOrganization = (
  id: string,
  name: string,
  credits: number,
  is_personal?: boolean,
): Organization => ({
  id,
  name,
  contact_name: "Contact Name",
  contact_email: "contact@example.com",
  conversation_expiration: 86400,
  agent: "default-agent",
  default_max_iterations: 20,
  security_analyzer: "standard",
  confirmation_mode: false,
  default_llm_model: "gpt-5-1",
  default_llm_api_key_for_byor: "*********",
  default_llm_base_url: "https://api.example-llm.com",
  remote_runtime_resource_factor: 2,
  enable_default_condenser: true,
  billing_margin: 0.15,
  enable_proactive_conversation_starters: true,
  sandbox_base_container_image: "ghcr.io/example/sandbox-base:latest",
  sandbox_runtime_container_image: "ghcr.io/example/sandbox-runtime:latest",
  org_version: 0,
  mcp_config: {
    tools: [],
    settings: {},
  },
  search_api_key: null,
  sandbox_api_key: null,
  max_budget_per_task: 25.0,
  enable_solvability_analysis: false,
  v1_enabled: true,
  credits,
  is_personal,
});

// Named mock organizations for test convenience
export const MOCK_PERSONAL_ORG = createMockOrganization(
  "1",
  "Personal Workspace",
  100,
  true,
);
export const MOCK_TEAM_ORG_ACME = createMockOrganization(
  "2",
  "Acme Corp",
  1000,
);
export const MOCK_TEAM_ORG_BETA = createMockOrganization("3", "Beta LLC", 500);
export const MOCK_TEAM_ORG_ALLHANDS = createMockOrganization(
  "4",
  "All Hands AI",
  750,
);

export const INITIAL_MOCK_ORGS: Organization[] = [
  MOCK_PERSONAL_ORG,
  MOCK_TEAM_ORG_ACME,
  MOCK_TEAM_ORG_BETA,
  MOCK_TEAM_ORG_ALLHANDS,
];

const INITIAL_MOCK_MEMBERS: Record<string, OrganizationMember[]> = {
  "1": [
    {
      org_id: "1",
      user_id: "99",
      email: "me@acme.org",
      role: "owner",
      llm_api_key: "**********",
      max_iterations: 20,
      llm_model: "gpt-4",
      llm_api_key_for_byor: null,
      llm_base_url: "https://api.openai.com",
      status: "active",
    },
  ],
  "2": [
    {
      org_id: "2",
      user_id: "1",
      email: "alice@acme.org",
      role: "owner",
      llm_api_key: "**********",
      max_iterations: 20,
      llm_model: "gpt-4",
      llm_api_key_for_byor: null,
      llm_base_url: "https://api.openai.com",
      status: "active",
    },
    {
      org_id: "1",
      user_id: "2",
      email: "bob@acme.org",
      role: "admin",
      llm_api_key: "**********",
      max_iterations: 20,
      llm_model: "gpt-4",
      llm_api_key_for_byor: null,
      llm_base_url: "https://api.openai.com",
      status: "active",
    },
    {
      org_id: "1",
      user_id: "3",
      email: "charlie@acme.org",
      role: "member",
      llm_api_key: "**********",
      max_iterations: 20,
      llm_model: "gpt-4",
      llm_api_key_for_byor: null,
      llm_base_url: "https://api.openai.com",
      status: "active",
    },
  ],
  "3": [
    {
      org_id: "2",
      user_id: "4",
      email: "tony@gamma.org",
      role: "member",
      llm_api_key: "**********",
      max_iterations: 20,
      llm_model: "gpt-4",
      llm_api_key_for_byor: null,
      llm_base_url: "https://api.openai.com",
      status: "active",
    },
    {
      org_id: "2",
      user_id: "5",
      email: "evan@gamma.org",
      role: "admin",
      llm_api_key: "**********",
      max_iterations: 20,
      llm_model: "gpt-4",
      llm_api_key_for_byor: null,
      llm_base_url: "https://api.openai.com",
      status: "active",
    },
  ],
  "4": [
    {
      org_id: "3",
      user_id: "6",
      email: "robert@all-hands.dev",
      role: "owner",
      llm_api_key: "**********",
      max_iterations: 20,
      llm_model: "gpt-4",
      llm_api_key_for_byor: null,
      llm_base_url: "https://api.openai.com",
      status: "active",
    },
    {
      org_id: "3",
      user_id: "7",
      email: "ray@all-hands.dev",
      role: "admin",
      llm_api_key: "**********",
      max_iterations: 20,
      llm_model: "gpt-4",
      llm_api_key_for_byor: null,
      llm_base_url: "https://api.openai.com",
      status: "active",
    },
    {
      org_id: "3",
      user_id: "8",
      email: "chuck@all-hands.dev",
      role: "member",
      llm_api_key: "**********",
      max_iterations: 20,
      llm_model: "gpt-4",
      llm_api_key_for_byor: null,
      llm_base_url: "https://api.openai.com",
      status: "active",
    },
    {
      org_id: "3",
      user_id: "9",
      email: "stephan@all-hands.dev",
      role: "member",
      llm_api_key: "**********",
      max_iterations: 20,
      llm_model: "gpt-4",
      llm_api_key_for_byor: null,
      llm_base_url: "https://api.openai.com",
      status: "active",
    },
    {
      org_id: "3",
      user_id: "10",
      email: "tim@all-hands.dev",
      role: "member",
      llm_api_key: "**********",
      max_iterations: 20,
      llm_model: "gpt-4",
      llm_api_key_for_byor: null,
      llm_base_url: "https://api.openai.com",
      status: "invited",
    },
  ],
};

export const ORGS_AND_MEMBERS: Record<string, OrganizationMember[]> = {
  "1": INITIAL_MOCK_MEMBERS["1"].map((member) => ({ ...member })),
  "2": INITIAL_MOCK_MEMBERS["2"].map((member) => ({ ...member })),
  "3": INITIAL_MOCK_MEMBERS["3"].map((member) => ({ ...member })),
  "4": INITIAL_MOCK_MEMBERS["4"].map((member) => ({ ...member })),
};

const orgs = new Map(INITIAL_MOCK_ORGS.map((org) => [org.id, org]));

export const resetOrgMockData = () => {
  // Reset organizations to initial state
  orgs.clear();
  INITIAL_MOCK_ORGS.forEach((org) => {
    orgs.set(org.id, { ...org });
  });
};

export const resetOrgsAndMembersMockData = () => {
  // Reset ORGS_AND_MEMBERS to initial state
  // Note: This is needed since ORGS_AND_MEMBERS is mutated by updateMember
  Object.keys(INITIAL_MOCK_MEMBERS).forEach((orgId) => {
    ORGS_AND_MEMBERS[orgId] = INITIAL_MOCK_MEMBERS[orgId].map((member) => ({
      ...member,
    }));
  });
};

export const ORG_HANDLERS = [
  http.get("/api/organizations/:orgId/me", ({ params }) => {
    const orgId = params.orgId?.toString();
    if (!orgId || !ORGS_AND_MEMBERS[orgId]) {
      return HttpResponse.json(
        { error: "Organization not found" },
        { status: 404 },
      );
    }

    let role: OrganizationUserRole = "member";
    switch (orgId) {
      case "1": // Personal Workspace
        role = "owner";
        break;
      case "2": // Acme Corp
        role = "owner";
        break;
      case "3": // Beta LLC
        role = "member";
        break;
      case "4": // All Hands AI
        role = "admin";
        break;
      default:
        role = "member";
    }

    const me: OrganizationMember = {
      ...MOCK_ME,
      org_id: orgId,
      role,
    };
    return HttpResponse.json(me);
  }),

  http.get("/api/organizations/:orgId/members", ({ params }) => {
    const orgId = params.orgId?.toString();
    if (!orgId || !ORGS_AND_MEMBERS[orgId]) {
      return HttpResponse.json(
        { error: "Organization not found" },
        { status: 404 },
      );
    }
    const members = ORGS_AND_MEMBERS[orgId];
    return HttpResponse.json(members);
  }),

  http.get("/api/organizations", () => {
    const organizations = Array.from(orgs.values());
    return HttpResponse.json({ items: organizations });
  }),

  http.patch("/api/organizations/:orgId", async ({ request, params }) => {
    const { name } = (await request.json()) as {
      name: string;
    };
    const orgId = params.orgId?.toString();

    if (!name) {
      return HttpResponse.json({ error: "Name is required" }, { status: 400 });
    }

    if (!orgId) {
      return HttpResponse.json(
        { error: "Organization ID is required" },
        { status: 400 },
      );
    }

    const existingOrg = orgs.get(orgId);
    if (!existingOrg) {
      return HttpResponse.json(
        { error: "Organization not found" },
        { status: 404 },
      );
    }

    const updatedOrg: Organization = {
      ...existingOrg,
      name,
    };
    orgs.set(orgId, updatedOrg);

    return HttpResponse.json(updatedOrg, { status: 201 });
  }),

  http.get("/api/organizations/:orgId", ({ params }) => {
    const orgId = params.orgId?.toString();

    if (orgId) {
      const org = orgs.get(orgId);
      if (org) return HttpResponse.json(org);
    }

    return HttpResponse.json(
      { error: "Organization not found" },
      { status: 404 },
    );
  }),

  http.delete("/api/organizations/:orgId", ({ params }) => {
    const orgId = params.orgId?.toString();

    if (orgId && orgs.has(orgId) && ORGS_AND_MEMBERS[orgId]) {
      orgs.delete(orgId);
      delete ORGS_AND_MEMBERS[orgId];
      return HttpResponse.json(
        { message: "Organization deleted" },
        { status: 204 },
      );
    }

    return HttpResponse.json(
      { error: "Organization not found" },
      { status: 404 },
    );
  }),

  http.get("/api/organizations/:orgId/payment", ({ params }) => {
    const orgId = params.orgId?.toString();

    if (orgId) {
      const org = orgs.get(orgId);
      if (org) {
        return HttpResponse.json({
          cardNumber: "**** **** **** 1234", // Mocked payment info
        });
      }
    }

    return HttpResponse.json(
      { error: "Organization not found" },
      { status: 404 },
    );
  }),

  http.patch(
    "/api/organizations/:orgId/members/:userId",
    async ({ request, params }) => {
      const updateData =
        (await request.json()) as UpdateOrganizationMemberParams;
      const orgId = params.orgId?.toString();
      const userId = params.userId?.toString();

      if (!orgId || !ORGS_AND_MEMBERS[orgId]) {
        return HttpResponse.json(
          { error: "Organization not found" },
          { status: 404 },
        );
      }

      const member = ORGS_AND_MEMBERS[orgId].find((m) => m.user_id === userId);
      if (!member) {
        return HttpResponse.json(
          { error: "Member not found" },
          { status: 404 },
        );
      }

      // Update member with any provided fields
      const newMember: OrganizationMember = {
        ...member,
        ...updateData,
      };
      const newMembers = ORGS_AND_MEMBERS[orgId].map((m) =>
        m.user_id === userId ? newMember : m,
      );
      ORGS_AND_MEMBERS[orgId] = newMembers;

      return HttpResponse.json(newMember, { status: 200 });
    },
  ),

  http.delete("/api/organizations/:orgId/members/:userId", ({ params }) => {
    const { orgId, userId } = params;

    if (!orgId || !userId || !ORGS_AND_MEMBERS[orgId as string]) {
      return HttpResponse.json(
        { error: "Organization or member not found" },
        { status: 404 },
      );
    }

    // Remove member from organization
    const members = ORGS_AND_MEMBERS[orgId as string];
    const updatedMembers = members.filter(
      (member) => member.user_id !== userId,
    );
    ORGS_AND_MEMBERS[orgId as string] = updatedMembers;

    return HttpResponse.json({ message: "Member removed" }, { status: 200 });
  }),

  http.post(
    "/api/organizations/:orgId/members/invite",
    async ({ request, params }) => {
      const { emails } = (await request.json()) as { emails: string[] };
      const orgId = params.orgId?.toString();

      if (!emails || emails.length === 0) {
        return HttpResponse.json(
          { error: "Emails are required" },
          { status: 400 },
        );
      }

      if (!orgId || !ORGS_AND_MEMBERS[orgId]) {
        return HttpResponse.json(
          { error: "Organization not found" },
          { status: 404 },
        );
      }

      const members = Array.from(ORGS_AND_MEMBERS[orgId]);
      const newMembers: OrganizationMember[] = emails.map((email, index) => ({
        org_id: orgId,
        user_id: String(members.length + index + 1),
        email,
        role: "member" as const,
        llm_api_key: "**********",
        max_iterations: 20,
        llm_model: "gpt-4",
        llm_api_key_for_byor: null,
        llm_base_url: "https://api.openai.com",
        status: "invited" as const,
      }));

      ORGS_AND_MEMBERS[orgId] = [...members, ...newMembers];

      return HttpResponse.json(newMembers, { status: 201 });
    },
  ),
];
