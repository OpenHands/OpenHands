import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ProjectPanel } from "#/components/features/projects/project-panel/project-panel";
import ProjectsService from "#/api/projects-service/projects-service.api";
import {
  BOARD,
  PROJECT_ID,
  RETRY,
  member,
  mockProjectServices,
} from "./project-fixtures";

const handleUpload = vi.fn();
vi.mock("#/hooks/chat/use-chat-attachment-upload", () => ({
  useChatAttachmentUpload: () => ({ handleUpload }),
}));

function renderPanel() {
  return render(
    <QueryClientProvider
      client={
        new QueryClient({ defaultOptions: { queries: { retry: false } } })
      }
    >
      <MemoryRouter>
        <ProjectPanel projectId={PROJECT_ID} />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("ProjectPanel", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    handleUpload.mockReset();
    mockProjectServices();
  });

  it("opens on Files and counts files and pull requests, each pull request once", async () => {
    renderPanel();
    await waitFor(() =>
      expect(screen.getByTestId("project-panel-tab-files")).toHaveTextContent(
        "PROJECTS$FILES · 3",
      ),
    );
    expect(screen.getByTestId("project-panel-tab-files")).toHaveAttribute(
      "aria-selected",
      "true",
    );
    expect(screen.getByTestId("project-panel-tab-prs")).toHaveTextContent(
      "PROJECTS$PRS · 3",
    );
    // The conversations live in the sidebar now.
    expect(screen.queryByTestId("project-panel-tab-conversations")).toBeNull();
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
    expect(within(open).getAllByTestId("project-panel-pr")).toHaveLength(1);
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
