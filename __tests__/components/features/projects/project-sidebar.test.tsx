import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { SidebarConversationList } from "#/components/features/sidebar/sidebar-conversation-list";
import { NavigationProvider } from "#/context/navigation-context";
import { experimentalFeaturesService } from "#/api/experimental-features-service/experimental-features-service.api";
import ProjectsService from "#/api/projects-service/projects-service.api";
import {
  AUDIT,
  BOARD,
  COORDINATOR,
  PROJECT_ID,
  RETRY,
  member,
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

  it("marks where the user is and shows the question it waits on", async () => {
    renderSidebar();
    const coordinator = await rowOf("PROJECTS$COORDINATOR");
    expect(coordinator).toHaveAttribute("aria-current", "page");
    expect(coordinator).toHaveTextContent("PROJECTS$YOU_ARE_HERE · ");
    expect(await rowOf("Rate-limit login")).toHaveTextContent(
      "PROJECTS$ASKS 5 or 10 tries per minute?",
    );
  });

  it("moves an idle conversation with a pull request ready for the user to Needs you", async () => {
    const docs = "bb000000000040008000000000000004";
    vi.spyOn(ProjectsService, "listConversations").mockResolvedValue({
      ...BOARD,
      conversations: [
        ...BOARD.conversations,
        member(docs, "Write login docs", {
          pull_requests: [{ owner: "acme", repo: "app", number: 412 }],
        }),
      ],
    });
    renderSidebar();
    const needsYou = await screen.findByTestId("project-panel-group-needs_you");
    await waitFor(() =>
      expect(needsYou).toHaveTextContent("PROJECTS$GROUP_NEEDS_YOU · 2"),
    );
    expect(await rowOf("Write login docs")).toHaveTextContent(
      "#412 · PROJECTS$PR_READY_HINT",
    );
    // A working conversation and a resolved one keep their place.
    expect(screen.getByTestId("project-panel-group-working")).toHaveTextContent(
      "Fix login retry",
    );
    expect(needsYou).not.toHaveTextContent("Audit auth flows");
  });

  it("chips each pull request by what it asks of the user, linked to GitHub", async () => {
    renderSidebar();
    const retry = (await rowOf("Fix login retry")).closest("li")!;
    const chip = (number: number) =>
      within(retry)
        .getAllByRole("link")
        .find((a) => a.getAttribute("href")?.endsWith(`/pull/${number}`))!;
    // #412 passes its checks, so it is ready for the user and comes first;
    // #409 could not be read.
    await waitFor(() =>
      expect(chip(412)).toHaveAttribute("data-attention", "ready"),
    );
    expect(chip(412)).toHaveTextContent("#412PROJECTS$PR_READY");
    expect(chip(412)).toHaveAttribute(
      "href",
      "https://github.com/acme/app/pull/412",
    );
    expect(chip(412)).toHaveAttribute("target", "_blank");
    expect(chip(409)).toHaveAttribute("data-attention", "unknown");
    expect(
      Array.from(
        retry.querySelectorAll('[data-testid="project-conversation-pr"]'),
      ).map((c) => c.textContent),
    ).toEqual(["#412PROJECTS$PR_READY", "#409"]);
    expect(
      retry.querySelector('[data-testid="project-conversation-age"]'),
    ).not.toBeNull();
    expect(
      retry.querySelector('[data-testid="project-conversation-files"]'),
    ).toHaveTextContent("PROJECTS$FILE_COUNT");
  });

  it("expands a folded group on client-side navigation while keeping manual fold on the same conversation", async () => {
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const tree = (path = `/conversations/${view.conversationId}`) => (
      <QueryClientProvider client={queryClient}>
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
      </QueryClientProvider>
    );

    const { rerender } = render(tree());
    const resolvedToggle = await screen.findByTestId(
      "project-panel-group-resolved-toggle",
    );
    expect(resolvedToggle).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByText("Audit auth flows")).toBeNull();

    // Navigating client-side to a resolved worker expands Resolved and marks it.
    openConversation(AUDIT, PROJECT_ID);
    rerender(tree());
    expect(resolvedToggle).toHaveAttribute("aria-expanded", "true");
    const auditRow = await rowOf("Audit auth flows");
    expect(auditRow).toHaveAttribute("aria-current", "page");
    expect(auditRow).toHaveTextContent("PROJECTS$YOU_ARE_HERE · ");

    // Folding Resolved by hand while staying on the same conversation keeps it folded.
    await userEvent.click(resolvedToggle);
    expect(resolvedToggle).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByText("Audit auth flows")).toBeNull();
    rerender(tree());
    expect(resolvedToggle).toHaveAttribute("aria-expanded", "false");

    // Navigating to a conversation in another manually folded group expands it too.
    const workingToggle = screen.getByTestId(
      "project-panel-group-working-toggle",
    );
    await userEvent.click(workingToggle);
    expect(workingToggle).toHaveAttribute("aria-expanded", "false");
    openConversation(RETRY, PROJECT_ID);
    rerender(tree());
    expect(workingToggle).toHaveAttribute("aria-expanded", "true");
    expect(await rowOf("Fix login retry")).toHaveAttribute(
      "aria-current",
      "page",
    );
  });
});
