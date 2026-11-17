import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { SidebarConversationList } from "#/components/features/sidebar/sidebar-conversation-list";
import { NavigationProvider } from "#/context/navigation-context";
import { experimentalFeaturesService } from "#/api/experimental-features-service/experimental-features-service.api";
import {
  AUDIT,
  COORDINATOR,
  PROJECT_ID,
  RETRY,
  mockProjectServices,
} from "./project-fixtures";

const view = vi.hoisted(() => ({
  conversationId: null as string | null,
  conversation: undefined as { tags?: Record<string, string> } | undefined,
}));
vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({ data: view.conversation }),
}));
vi.mock("#/components/features/conversation-panel/conversation-panel", () => ({
  ConversationPanel: () => <div data-testid="conversation-panel" />,
}));

/** Opens a conversation; `project` is its project tag, if any. */
function openConversation(id: string, project?: string) {
  view.conversationId = id;
  view.conversation = { tags: project ? { project } : {} };
}

function renderSidebar(path = `/conversations/${view.conversationId}`) {
  return render(
    <QueryClientProvider
      client={
        new QueryClient({ defaultOptions: { queries: { retry: false } } })
      }
    >
      <MemoryRouter initialEntries={[path]}>
        <NavigationProvider
          value={{
            currentPath: path,
            conversationId: view.conversationId,
            isNavigating: false,
            navigate: () => {},
          }}
        >
          <SidebarConversationList collapsed={false} />
        </NavigationProvider>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

const rowOf = async (title: string) =>
  (await screen.findAllByTestId("project-conversation")).find((li) =>
    li.textContent?.includes(title),
  )!;

describe("Sidebar in a project view", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    mockProjectServices();
    openConversation(COORDINATOR, PROJECT_ID);
  });

  it("replaces the conversation list with the project's conversations", async () => {
    renderSidebar();
    expect(await screen.findByTestId("project-sidebar")).toBeInTheDocument();
    expect(screen.queryByTestId("conversation-panel")).toBeNull();
    await waitFor(() =>
      expect(screen.getByTestId("project-sidebar-name")).toHaveTextContent(
        "Login hardening",
      ),
    );
    expect(screen.getByTestId("project-sidebar-name")).toHaveAttribute(
      "href",
      `/projects/${PROJECT_ID}`,
    );
  });

  it("does the same on the project's page, with nothing marked as open", async () => {
    view.conversationId = null;
    view.conversation = undefined;
    renderSidebar(`/projects/${PROJECT_ID}`);
    expect(await screen.findByTestId("project-sidebar-name")).toHaveAttribute(
      "aria-current",
      "page",
    );
    expect(await rowOf("PROJECTS$COORDINATOR")).not.toHaveAttribute(
      "aria-current",
    );
  });

  it("keeps the full list outside a project, or with Projects turned off", async () => {
    openConversation(RETRY);
    const { unmount } = renderSidebar();
    expect(await screen.findByTestId("conversation-panel")).toBeInTheDocument();
    unmount();

    vi.mocked(
      experimentalFeaturesService.listExperimentalFeatures,
    ).mockResolvedValue([]);
    openConversation(RETRY, PROJECT_ID);
    renderSidebar();
    expect(await screen.findByTestId("conversation-panel")).toBeInTheDocument();
    expect(screen.queryByTestId("project-sidebar")).toBeNull();
  });

  it("pins the coordinator, groups the rest like the board and folds Resolved", async () => {
    renderSidebar();
    const rows = await screen.findAllByTestId("project-conversation");
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

  it("opens Resolved when the open conversation is in it", async () => {
    openConversation(AUDIT, PROJECT_ID);
    renderSidebar();
    expect(await rowOf("Audit auth flows")).toHaveAttribute(
      "aria-current",
      "page",
    );
  });

  it("marks where the user is and shows the question before the latest line", async () => {
    renderSidebar();
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
    renderSidebar();
    const retry = await rowOf("Fix login retry");
    const chips = Array.from(
      retry.querySelectorAll('[data-testid="project-conversation-pr"]'),
    );
    expect(chips.map((c) => c.textContent)).toEqual(["#412", "#409"]);
    await waitFor(() =>
      expect(chips[0].querySelector('[data-checks="success"]')).not.toBeNull(),
    );
    expect(
      retry.querySelector('[data-testid="project-conversation-files"]'),
    ).toHaveTextContent("PROJECTS$FILE_COUNT");
  });
});
