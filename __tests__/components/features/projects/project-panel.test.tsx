import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ProjectPanel } from "#/components/features/projects/project-panel/project-panel";
import ProjectsService, {
  type ProjectArtifact,
  type ProjectConversation,
} from "#/api/projects-service/projects-service.api";
import * as pullRequests from "#/api/git-service/pull-request-status.api";
import { experimentalFeaturesService } from "#/api/experimental-features-service/experimental-features-service.api";
import { getConversationState } from "#/utils/conversation-local-storage";

const handleUpload = vi.fn();
vi.mock("#/hooks/chat/use-chat-attachment-upload", () => ({
  useChatAttachmentUpload: () => ({ handleUpload }),
}));

const PROJECT_ID = "aa000000000040008000000000000001";
const COORDINATOR = "cc000000000040008000000000000001";
const RETRY = "bb000000000040008000000000000001";
const LIMITS = "bb000000000040008000000000000002";
const AUDIT = "bb000000000040008000000000000003";
const NOW = "2026-11-17T00:00:00Z";

const member = (
  id: string,
  title: string,
  extra: Partial<ProjectConversation> = {},
): ProjectConversation => ({
  conversation_id: id,
  title,
  execution_status: "finished",
  attention: "idle",
  resolved: null,
  turn: "answered",
  ask: null,
  last_message: null,
  pull_requests: [],
  automation_id: null,
  parent_conversation_id: COORDINATOR,
  created_at: NOW,
  updated_at: NOW,
  ...extra,
});

const BOARD = {
  project_id: PROJECT_ID,
  coordinator: member(COORDINATOR, "Login hardening", {
    parent_conversation_id: null,
    last_message: "Started two conversations.",
  }),
  conversations: [
    member(LIMITS, "Rate-limit login", {
      attention: "needs_input",
      turn: "asked",
      ask: "5 or 10 tries per minute?",
      last_message: "Here is the comparison.",
    }),
    member(RETRY, "Fix login retry", {
      attention: "working",
      execution_status: "running",
      last_message: "Opened the pull request.",
      pull_requests: [
        { owner: "acme", repo: "app", number: 412 },
        { owner: "acme", repo: "app", number: 409 },
      ],
    }),
    member(AUDIT, "Audit auth flows", {
      attention: "resolved",
      resolved: "user",
      pull_requests: [{ owner: "acme", repo: "lib", number: 9 }],
    }),
  ],
};

const artifact = (
  name: string,
  extra: Partial<ProjectArtifact> = {},
): ProjectArtifact => ({
  name,
  size: 10,
  source: "agent",
  conversation_id: RETRY,
  created_at: NOW,
  updated_at: NOW,
  ...extra,
});

const ARTIFACTS = [
  artifact("retry-design.md"),
  artifact("backoff.txt"),
  artifact("threat-model.md", { source: "user", conversation_id: null }),
];

function renderPanel(conversationId = COORDINATOR) {
  return render(
    <QueryClientProvider
      client={
        new QueryClient({ defaultOptions: { queries: { retry: false } } })
      }
    >
      <MemoryRouter>
        <ProjectPanel projectId={PROJECT_ID} conversationId={conversationId} />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

const rowOf = async (title: string) =>
  (await screen.findAllByTestId("project-panel-conversation")).find((li) =>
    li.textContent?.includes(title),
  )!;

describe("ProjectPanel", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    localStorage.clear();
    handleUpload.mockReset();
    vi.spyOn(
      experimentalFeaturesService,
      "listExperimentalFeatures",
    ).mockResolvedValue([
      { key: "projects", label: "Projects", description: "", enabled: true },
    ] as Awaited<
      ReturnType<typeof experimentalFeaturesService.listExperimentalFeatures>
    >);
    vi.spyOn(ProjectsService, "getProject").mockResolvedValue({
      id: PROJECT_ID,
      name: "Login hardening",
      description: "",
      coordinator_conversation_id: COORDINATOR,
      worker_agent_profile_id: null,
      created_at: NOW,
      updated_at: NOW,
    });
    vi.spyOn(ProjectsService, "listConversations").mockResolvedValue(BOARD);
    vi.spyOn(ProjectsService, "listArtifacts").mockResolvedValue(ARTIFACTS);
    vi.spyOn(ProjectsService, "readArtifactText").mockResolvedValue("text");
    vi.spyOn(ProjectsService, "downloadArtifact").mockResolvedValue(
      new Blob(["text"], { type: "text/plain" }),
    );
    vi.spyOn(pullRequests, "getPullRequestStatus").mockImplementation(
      async (owner, repo, number) => {
        if (number === 409) throw new Error("409 missing_provider_token");
        return {
          repository: `${owner}/${repo}`,
          number,
          title: number === 412 ? "Retry login on network errors" : "Lib fix",
          url: `https://github.com/${owner}/${repo}/pull/${number}`,
          state: number === 412 ? "open" : "merged",
          checks: number === 412 ? "success" : "none",
        };
      },
    );
  });

  it("names the project, counts files and pull requests, and links the project page", async () => {
    renderPanel();
    await waitFor(() =>
      expect(screen.getByTestId("project-panel-name")).toHaveTextContent(
        "Login hardening",
      ),
    );
    await waitFor(() =>
      expect(screen.getByTestId("project-panel-tab-files")).toHaveTextContent(
        "PROJECTS$FILES · 3",
      ),
    );
    expect(screen.getByTestId("project-panel-tab-prs")).toHaveTextContent(
      "PROJECTS$PRS · 3",
    );
    expect(screen.getByTestId("project-panel-open-project")).toHaveAttribute(
      "href",
      `/projects/${PROJECT_ID}`,
    );
  });

  it("pins the coordinator, groups the rest like the board and folds Resolved", async () => {
    renderPanel();
    const rows = await screen.findAllByTestId("project-panel-conversation");
    expect(rows.map((r) => r.querySelector(".truncate")?.textContent)).toEqual([
      "PROJECTS$COORDINATOR",
      "Rate-limit login",
      "Fix login retry",
    ]);
    expect(
      screen.getByTestId("project-panel-group-needs_you"),
    ).toHaveTextContent("PROJECTS$GROUP_NEEDS_YOU · 1");
    expect(screen.queryByTestId("project-panel-group-idle")).toBeNull();

    await userEvent.click(
      screen.getByTestId("project-panel-group-resolved-toggle"),
    );
    expect(await rowOf("Audit auth flows")).toBeInTheDocument();
  });

  it("marks where the user is and shows the question before the latest line", async () => {
    renderPanel(COORDINATOR);
    const coordinator = await rowOf("PROJECTS$COORDINATOR");
    expect(coordinator).toHaveAttribute("aria-current", "page");
    expect(coordinator).toHaveTextContent(
      "PROJECTS$YOU_ARE_HERE · Started two conversations.",
    );
    expect(await rowOf("Rate-limit login")).toHaveTextContent(
      "PROJECTS$ASKS 5 or 10 tries per minute?",
    );
    expect(await rowOf("Rate-limit login")).not.toHaveTextContent(
      "Here is the comparison.",
    );
    expect(await rowOf("Fix login retry")).toHaveTextContent(
      "Opened the pull request.",
    );
  });

  it("chips each conversation's pull requests with their checks, and its files", async () => {
    renderPanel();
    const retry = await rowOf("Fix login retry");
    const chips = within(retry).getAllByTestId("project-panel-conversation-pr");
    expect(chips.map((c) => c.textContent)).toEqual(["#412", "#409"]);
    await waitFor(() =>
      expect(chips[0].querySelector('[data-checks="success"]')).not.toBeNull(),
    );
    expect(
      within(retry).getByTestId("project-panel-conversation-files"),
    ).toHaveTextContent("PROJECTS$FILE_COUNT");
  });

  it("keeps the panel on the Project tab when it opens another conversation", async () => {
    renderPanel();
    await userEvent.click(await rowOf("Fix login retry"));
    expect(getConversationState(RETRY).selectedTab).toBe("project");
  });

  it("lists pull requests by state, newest first, with who opened them", async () => {
    renderPanel();
    await userEvent.click(await screen.findByTestId("project-panel-tab-prs"));
    await waitFor(() =>
      expect(
        screen.getByTestId("project-panel-group-prs-open"),
      ).toHaveTextContent("Retry login on network errors"),
    );
    const open = screen.getByTestId("project-panel-group-prs-open");
    expect(within(open).getByTestId("project-panel-pr")).toHaveAttribute(
      "href",
      "https://github.com/acme/app/pull/412",
    );
    expect(open).toHaveTextContent("acme/app#412 · Fix login retry");
    expect(
      screen.getByTestId("project-panel-group-prs-merged"),
    ).toHaveTextContent("acme/lib#9 · Audit auth flows");
    // Without a provider token the state is unknown, but the link stays.
    const unknown = screen.getByTestId("project-panel-group-prs-unknown");
    expect(within(unknown).getByTestId("project-panel-pr")).toHaveAttribute(
      "href",
      "https://github.com/acme/app/pull/409",
    );
  });

  it("says so when no conversation opened a pull request", async () => {
    vi.mocked(ProjectsService.listConversations).mockResolvedValue({
      ...BOARD,
      conversations: [member(RETRY, "Fix login retry")],
    });
    renderPanel();
    await userEvent.click(await screen.findByTestId("project-panel-tab-prs"));
    expect(
      await screen.findByTestId("project-panel-prs-empty"),
    ).toBeInTheDocument();
  });

  it("splits files by who made them, previews in place and attaches to the message", async () => {
    renderPanel();
    await userEvent.click(await screen.findByTestId("project-panel-tab-files"));
    const agents = await screen.findByTestId(
      "project-panel-group-files-agents",
    );
    expect(agents).toHaveTextContent("retry-design.md");
    expect(agents).toHaveTextContent("Fix login retry");
    expect(
      screen.getByTestId("project-panel-group-files-user"),
    ).toHaveTextContent("threat-model.md");

    const [first] = within(agents).getAllByTestId("project-panel-file");
    await userEvent.click(within(first).getAllByRole("button")[0]);
    expect(
      await within(first).findByTestId("project-file-preview"),
    ).toHaveTextContent("text");

    await userEvent.click(
      within(first).getByTestId("project-panel-file-attach"),
    );
    await waitFor(() => expect(handleUpload).toHaveBeenCalledTimes(1));
    const [[files]] = handleUpload.mock.calls;
    expect(files[0].name).toBe("retry-design.md");
  });
});
