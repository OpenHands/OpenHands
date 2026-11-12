import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ProjectPage from "#/routes/project";
import ProjectsService, {
  type ProjectConversation,
} from "#/api/projects-service/projects-service.api";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { experimentalFeaturesService } from "#/api/experimental-features-service/experimental-features-service.api";
import { I18nKey } from "#/i18n/declaration";

const PROJECT_ID = "aa000000000040008000000000000001";
const COORDINATOR_ID = "cc000000000040008000000000000001";

const member = (
  id: string,
  title: string,
  attention: ProjectConversation["attention"],
  extra: Partial<ProjectConversation> = {},
): ProjectConversation => ({
  conversation_id: id,
  title,
  execution_status: "finished",
  attention,
  resolved: attention === "resolved" ? "user" : null,
  turn: null,
  ask: null,
  automation_id: null,
  parent_conversation_id: COORDINATOR_ID,
  created_at: "2026-11-09T10:00:00Z",
  updated_at: "2026-11-09T10:00:00Z",
  ...extra,
});

const setProjectsFeature = (enabled: boolean) =>
  vi
    .spyOn(experimentalFeaturesService, "listExperimentalFeatures")
    .mockResolvedValue([
      {
        key: "projects",
        label: "Projects",
        description: "",
        enabled,
      },
    ] as Awaited<
      ReturnType<typeof experimentalFeaturesService.listExperimentalFeatures>
    >);

function renderPage() {
  return render(
    <QueryClientProvider
      client={
        new QueryClient({ defaultOptions: { queries: { retry: false } } })
      }
    >
      <MemoryRouter initialEntries={[`/projects/${PROJECT_ID}`]}>
        <Routes>
          <Route path="/projects/:projectId" element={<ProjectPage />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("ProjectPage", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(ProjectsService, "getProject").mockResolvedValue({
      id: PROJECT_ID,
      name: "Insider 0.2",
      description: "Get Insider ready for people to try",
      coordinator_conversation_id: COORDINATOR_ID,
      worker_agent_profile_id: null,
      created_at: "2026-11-09T09:00:00Z",
      updated_at: "2026-11-09T09:00:00Z",
    });
    vi.spyOn(ProjectsService, "listConversations").mockResolvedValue({
      project_id: PROJECT_ID,
      coordinator: member(COORDINATOR_ID, "Coordinator", "idle"),
      conversations: [
        member("01000000000040008000000000000001", "Changelog", "needs_input", {
          turn: "asked",
          ask: "Install from main or a branch?",
        }),
        member("02000000000040008000000000000001", "README", "working", {
          execution_status: "running",
        }),
        member("03000000000040008000000000000001", "Locales", "resolved"),
        member("06000000000040008000000000000001", "Docs", "idle"),
      ],
    });
    vi.spyOn(ProjectsService, "listArtifacts").mockResolvedValue([
      {
        name: "notes.md",
        size: 5,
        source: "user",
        conversation_id: null,
        created_at: "2026-11-09T10:00:00Z",
        updated_at: "2026-11-09T10:00:00Z",
      },
      {
        name: "CHANGELOG.md",
        size: 5,
        source: "agent",
        conversation_id: "01000000000040008000000000000001",
        created_at: "2026-11-09T11:00:00Z",
        updated_at: "2026-11-09T11:00:00Z",
      },
    ]);
    vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    ).mockResolvedValue({ items: [], next_page_id: null });
  });

  it("says how to turn Projects on while the flag is off", async () => {
    setProjectsFeature(false);
    renderPage();
    expect(await screen.findByTestId("projects-off")).toBeInTheDocument();
    expect(ProjectsService.getProject).not.toHaveBeenCalled();
  });

  it("shows the coordinator, the board grouped by what needs you, and files", async () => {
    setProjectsFeature(true);
    renderPage();

    expect(await screen.findByTestId("project-name")).toHaveTextContent(
      "Insider 0.2",
    );
    const coordinator = screen.getByTestId("project-open-coordinator");
    expect(coordinator).toHaveAttribute(
      "href",
      expect.stringContaining(`/conversations/${COORDINATOR_ID}`),
    );

    const needsYou = await screen.findByTestId("project-group-needs_you");
    expect(within(needsYou).getByText("Changelog")).toBeInTheDocument();
    expect(needsYou).toHaveTextContent("Install from main or a branch?");
    expect(
      within(screen.getByTestId("project-group-working")).getByText("README"),
    ).toBeInTheDocument();
    expect(
      within(screen.getByTestId("project-group-resolved")).getByText("Locales"),
    ).toBeInTheDocument();
    // Not running, not waiting on the user, not resolved: Idle, not "Done".
    const idle = screen.getByTestId("project-group-idle");
    expect(within(idle).getByText("Docs")).toBeInTheDocument();
    expect(idle).toHaveTextContent(I18nKey.PROJECTS$GROUP_IDLE);

    const files = await screen.findAllByTestId("project-file");
    expect(files.map((f) => f.getAttribute("data-source"))).toEqual([
      "agent",
      "user",
    ]);
    // An agent's file names the conversation that saved it.
    expect(within(files[0]).getByText("Changelog")).toBeInTheDocument();
  });

  it.each([
    ["project-group-working", "02000000000040008000000000000001", true],
    ["project-group-resolved", "03000000000040008000000000000001", false],
  ])(
    "resolves or reopens a card in %s through the project's route",
    async (group, conversationId, resolved) => {
      setProjectsFeature(true);
      const route = vi
        .spyOn(ProjectsService, "setConversationResolved")
        .mockResolvedValue(undefined);
      const replace = vi.spyOn(
        AgentServerConversationService,
        "replaceConversationTags",
      );
      renderPage();

      const column = await screen.findByTestId(group);
      await userEvent.click(within(column).getByTestId("project-card-resolve"));
      await waitFor(() =>
        expect(route).toHaveBeenCalledWith(
          PROJECT_ID,
          conversationId,
          resolved,
        ),
      );
      expect(replace).not.toHaveBeenCalled();
    },
  );

  it("says why the server resolved a card on its own", async () => {
    setProjectsFeature(true);
    vi.spyOn(ProjectsService, "listConversations").mockResolvedValue({
      project_id: PROJECT_ID,
      coordinator: member(COORDINATOR_ID, "Coordinator", "idle"),
      conversations: [
        member("04000000000040008000000000000001", "Answer", "resolved", {
          resolved: "auto",
          resolve_note: "0.93 nothing",
        }),
        member("05000000000040008000000000000001", "Mine", "resolved"),
      ],
    });
    renderPage();

    const resolved = await screen.findByTestId("project-group-resolved");
    const notes = within(resolved).getAllByTestId("project-card-auto-resolved");
    expect(notes).toHaveLength(1);
    expect(notes[0]).toHaveTextContent(I18nKey.PROJECTS$AUTO_RESOLVED);
  });
});
