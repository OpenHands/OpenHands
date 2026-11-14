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
import { useCreateConversation } from "#/hooks/mutation/use-create-conversation";

vi.mock("#/hooks/mutation/use-create-conversation", () => ({
  useCreateConversation: vi.fn(() => ({ mutateAsync: vi.fn() })),
}));

const PROJECT_ID = "aa000000000040008000000000000001";
const COORDINATOR_ID = "cc000000000040008000000000000001";
const NOW = "2026-11-09T10:00:00Z";

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
    vi.spyOn(ProjectsService, "listMemory").mockResolvedValue([
      { name: "MEMORY.md", size: 12, revision: "r1", updated_at: NOW },
    ]);
    vi.spyOn(ProjectsService, "readMemory").mockResolvedValue(
      "- Releases are cut from main.",
    );
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
  it("says what each card under Needs you wants, in words", async () => {
    setProjectsFeature(true);
    vi.spyOn(ProjectsService, "listConversations").mockResolvedValue({
      project_id: PROJECT_ID,
      coordinator: member(COORDINATOR_ID, "Coordinator", "needs_you", {
        execution_status: "waiting_for_confirmation",
      }),
      conversations: [
        member("01000000000040008000000000000001", "Asks", "needs_input", {
          turn: "asked",
          ask: "Which branch?",
        }),
        member("02000000000040008000000000000001", "Approve", "needs_you", {
          execution_status: "waiting_for_confirmation",
        }),
        member("03000000000040008000000000000001", "Broke", "needs_you", {
          execution_status: "error",
        }),
      ],
    });
    renderPage();

    const needsYou = await screen.findByTestId("project-group-needs_you");
    const states = within(needsYou)
      .getAllByTestId("project-card-state")
      .map((el) => el.textContent);
    expect(states).toEqual([
      I18nKey.PROJECTS$STATE_ASKED,
      I18nKey.PROJECTS$STATE_WAITING_APPROVAL,
      I18nKey.PROJECTS$STATE_ERROR,
    ]);
    expect(needsYou).not.toHaveTextContent("waiting_for_confirmation");

    // The coordinator's own state shows on the page, not only its link.
    const coordinatorState = screen.getByTestId("project-coordinator-state");
    expect(coordinatorState).toHaveTextContent(
      I18nKey.PROJECTS$STATE_WAITING_APPROVAL,
    );
    expect(coordinatorState).toHaveAttribute("data-attention", "needs_you");
  });

  it("asks before removing a conversation from the project", async () => {
    setProjectsFeature(true);
    const remove = vi
      .spyOn(ProjectsService, "removeConversation")
      .mockResolvedValue(undefined);
    renderPage();

    const idle = await screen.findByTestId("project-group-idle");
    await userEvent.click(within(idle).getByTestId("project-card-remove"));
    expect(remove).not.toHaveBeenCalled();

    await userEvent.click(screen.getByTestId("cancel-button"));
    expect(screen.queryByTestId("confirmation-modal")).not.toBeInTheDocument();
    expect(remove).not.toHaveBeenCalled();

    await userEvent.click(within(idle).getByTestId("project-card-remove"));
    await userEvent.click(screen.getByTestId("confirm-button"));
    await waitFor(() =>
      expect(remove).toHaveBeenCalledWith(
        PROJECT_ID,
        "06000000000040008000000000000001",
      ),
    );
  });

  it("shows the project's memory and opens a file", async () => {
    setProjectsFeature(true);
    renderPage();

    const memory = await screen.findByTestId("project-memory");
    await userEvent.click(await within(memory).findByText("MEMORY.md"));
    expect(
      await within(memory).findByTestId("project-memory-preview"),
    ).toHaveTextContent("Releases are cut from main.");
    expect(ProjectsService.readMemory).toHaveBeenCalledWith(
      PROJECT_ID,
      "MEMORY.md",
    );
  });

  it("renames a project", async () => {
    setProjectsFeature(true);
    const update = vi
      .spyOn(ProjectsService, "updateProject")
      .mockImplementation(async (id, input) => ({
        id,
        name: input.name ?? "",
        description: input.description ?? "",
        coordinator_conversation_id: COORDINATOR_ID,
        worker_agent_profile_id: null,
        created_at: NOW,
        updated_at: NOW,
      }));
    renderPage();

    await userEvent.click(await screen.findByTestId("project-edit"));
    const name = screen.getByTestId("edit-project-name");
    await userEvent.clear(name);
    await userEvent.type(name, "Insider 0.3");
    await userEvent.click(screen.getByTestId("edit-project-save"));

    await waitFor(() =>
      expect(update).toHaveBeenCalledWith(PROJECT_ID, {
        name: "Insider 0.3",
        description: "Get Insider ready for people to try",
      }),
    );
    await waitFor(() =>
      expect(
        screen.queryByTestId("edit-project-modal"),
      ).not.toBeInTheDocument(),
    );
  });

  it("deletes a project only after a confirmation", async () => {
    setProjectsFeature(true);
    const remove = vi
      .spyOn(ProjectsService, "deleteProject")
      .mockResolvedValue(undefined);
    renderPage();

    await userEvent.click(await screen.findByTestId("project-edit"));
    await userEvent.click(screen.getByTestId("edit-project-delete"));
    expect(remove).not.toHaveBeenCalled();
    await userEvent.click(screen.getByTestId("confirm-button"));
    await waitFor(() => expect(remove).toHaveBeenCalledWith(PROJECT_ID));
  });
  it("starts a new conversation inside the project", async () => {
    setProjectsFeature(true);
    vi.mocked(useCreateConversation).mockReturnValue({
      mutateAsync: vi.fn().mockResolvedValue({
        conversation_id: "09000000000040008000000000000001",
        session_api_key: null,
        url: null,
      }),
    } as unknown as ReturnType<typeof useCreateConversation>);
    const add = vi
      .spyOn(ProjectsService, "addConversation")
      .mockResolvedValue(undefined);
    renderPage();

    await userEvent.click(
      await screen.findByTestId("project-new-conversation"),
    );
    await waitFor(() =>
      expect(add).toHaveBeenCalledWith(
        PROJECT_ID,
        "09000000000040008000000000000001",
      ),
    );
  });
});
