import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AgentOrganizationChart } from "#/components/features/organization/agent-organization-chart";
import AgentProfilesService, {
  type AgentProfileSummary,
} from "#/api/agent-profiles-service/agent-profiles-service.api";
import { ORGANIZATION_ROLES } from "#/constants/agent-organization";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, params?: Record<string, string>) =>
      params ? `${key}:${Object.values(params).join(",")}` : key,
  }),
}));

vi.mock("#/api/agent-profiles-service/agent-profiles-service.api");
vi.mock("#/api/settings-service/settings-service.api");
vi.mock("#/utils/custom-toast-handlers");
vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({
    backend: { id: "b1", kind: "local" },
    orgId: null,
  }),
}));

const navigate = vi.fn();
vi.mock("#/context/navigation-context", () => ({
  useNavigation: () => ({ navigate }),
}));

function summary(
  id: string,
  name: string,
  llmProfileRef: string | null = "gpt",
): AgentProfileSummary {
  return {
    id,
    name,
    agent_kind: "openhands",
    revision: 0,
    llm_profile_ref: llmProfileRef,
    mcp_server_refs: null,
  };
}

function renderChart() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <AgentOrganizationChart />
    </QueryClientProvider>,
  );
}

const department = (name: string) => screen.getByRole("region", { name });

describe("AgentOrganizationChart", () => {
  beforeEach(() => {
    vi.mocked(AgentProfilesService.listProfiles).mockResolvedValue({
      profiles: [
        summary("id-default", "default"),
        summary("id-lead", "leadership.tech-lead"),
        summary("id-backend", "engineering.backend"),
        summary("id-custom", "engineering.payments-bot"),
      ],
      active_agent_profile_id: "id-default",
    });
    vi.mocked(AgentProfilesService.saveProfile).mockResolvedValue({
      name: "x",
    } as never);
    vi.mocked(AgentProfilesService.activateProfile).mockResolvedValue(
      {} as never,
    );
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("places each agent in the department named by its prefix", async () => {
    renderChart();

    await screen.findByText("ORG$ROLE_TECH_LEAD");
    expect(
      within(department("ORG$DEPT_LEADERSHIP")).getByText("ORG$ROLE_TECH_LEAD"),
    ).toBeInTheDocument();
    const engineering = department("ORG$DEPT_ENGINEERING");
    expect(
      within(engineering).getByText("ORG$ROLE_BACKEND"),
    ).toBeInTheDocument();
    // A role with no template keeps its own name.
    expect(within(engineering).getByText("payments-bot")).toBeInTheDocument();
    expect(
      within(department("ORG$UNASSIGNED")).getByText("default"),
    ).toBeInTheDocument();
    expect(
      within(department("ORG$DEPT_QUALITY")).getByText("ORG$EMPTY_DEPARTMENT"),
    ).toBeInTheDocument();
  });

  it("offers only the template roles a department is still missing", async () => {
    const user = userEvent.setup();
    renderChart();
    await screen.findByText("ORG$ROLE_BACKEND");

    await user.click(
      within(department("ORG$DEPT_ENGINEERING")).getByRole("button", {
        name: "ORG$ADD_ROLE",
      }),
    );

    const menu = screen.getByRole("menu");
    expect(
      within(menu).getByRole("menuitem", { name: /ORG\$ROLE_FRONTEND/ }),
    ).toBeInTheDocument();
    expect(
      within(menu).queryByRole("menuitem", { name: /ORG\$ROLE_BACKEND/ }),
    ).not.toBeInTheDocument();
  });

  it("creates a role as an agent profile on the active agent's LLM with role instructions", async () => {
    const user = userEvent.setup();
    renderChart();
    await screen.findByText("ORG$ROLE_BACKEND");
    const qa = ORGANIZATION_ROLES.find((role) => role.id === "qa")!;

    await user.click(
      within(department("ORG$DEPT_QUALITY")).getByRole("button", {
        name: "ORG$ADD_ROLE",
      }),
    );
    await user.click(
      screen.getByRole("menuitem", {
        name: new RegExp(qa.titleKey.replace("$", "\\$")),
      }),
    );

    await waitFor(() =>
      expect(AgentProfilesService.saveProfile).toHaveBeenCalledWith(
        "quality.qa",
        {
          agent_kind: "openhands",
          llm_profile_ref: "gpt",
          system_message_suffix: qa.instructions,
        },
      ),
    );
  });

  it("explains that an LLM is needed when the active agent has none", async () => {
    vi.mocked(AgentProfilesService.listProfiles).mockResolvedValue({
      profiles: [summary("id-default", "default", null)],
      active_agent_profile_id: "id-default",
    });
    const user = userEvent.setup();
    renderChart();
    await screen.findByText("default");

    await user.click(
      within(department("ORG$DEPT_QUALITY")).getByRole("button", {
        name: "ORG$ADD_ROLE",
      }),
    );
    await user.click(screen.getAllByRole("menuitem")[0]);

    expect(displayErrorToast).toHaveBeenCalledWith(
      "SETTINGS$AGENT_PROFILE_NO_LLM",
    );
    expect(AgentProfilesService.saveProfile).not.toHaveBeenCalled();
  });

  it("activates an agent and opens a new chat when Chat is chosen", async () => {
    const user = userEvent.setup();
    renderChart();
    await screen.findByText("ORG$ROLE_BACKEND");

    await user.click(
      screen.getByRole("button", {
        name: "ORG$CHAT_WITH:ORG$ROLE_BACKEND",
      }),
    );

    await waitFor(() =>
      expect(AgentProfilesService.activateProfile).toHaveBeenCalledWith(
        "id-backend",
      ),
    );
    await waitFor(() =>
      expect(navigate).toHaveBeenCalledWith("/conversations"),
    );
  });
});
