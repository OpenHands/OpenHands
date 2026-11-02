import React from "react";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { compile } from "@tailwindcss/node";
import postcss, { type Container, type Document, type Rule } from "postcss";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  GitProviderPullRequestStatusService,
  type ConversationPullRequest,
} from "#/api/git-provider-pull-request-status-service";
import { ConversationOverviewDrawer } from "#/components/features/conversation/conversation-overview-drawer";
import { CONVERSATION_OVERVIEW_DRAWER_WIDTH_STORAGE_KEY } from "#/components/features/conversation/conversation-overview-drawer.constants";
import { ConversationOverviewDrawerContent } from "#/components/features/conversation/conversation-overview-drawer-content";
import {
  ConversationOverviewDrawerProvider,
  useConversationOverviewDrawer,
} from "#/components/features/conversation/conversation-overview-drawer-context";
import { CONVERSATION_OVERVIEW_DRAWER_SECTION } from "#/components/features/conversation/conversation-overview-drawer.types";
import { useCloseOnEscape } from "#/hooks/use-close-on-escape";

vi.mock("#/hooks/use-conversation-overview-stats", () => ({
  useConversationOverviewStats: () => ({
    workspaceName: "demo",
  }),
}));

vi.mock("#/hooks/use-conversation-primary-repository", () => ({
  useConversationPrimaryRepository: () => ({
    repository: "openhands/agent-canvas",
    provider: "github" as const,
    branch: "main",
    isConnected: true,
  }),
}));

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({ data: { id: "conv-1", pr_number: [] } }),
}));

vi.mock("#/hooks/query/use-local-git-info", () => ({
  useLocalGitInfo: () => ({ data: undefined }),
}));

vi.mock("#/hooks/query/use-repository-git-items", () => ({
  useRepositoryIssues: () => ({
    data: [],
    isLoading: false,
    isError: false,
  }),
}));

function OpenSection({
  section,
}: {
  section: (typeof CONVERSATION_OVERVIEW_DRAWER_SECTION)[keyof typeof CONVERSATION_OVERVIEW_DRAWER_SECTION];
}) {
  const { openSection } = useConversationOverviewDrawer();
  return (
    <button
      type="button"
      data-testid="open-drawer-section"
      onClick={() => openSection(section)}
    >
      Open
    </button>
  );
}

// Stands in for a menu that closes on Escape, such as the Overview ⋯ menu.
function MenuOverDrawer() {
  const [isOpen, setIsOpen] = React.useState(false);
  useCloseOnEscape(isOpen, () => setIsOpen(false));
  return (
    <>
      <button
        type="button"
        data-testid="open-menu-over-drawer"
        onClick={() => setIsOpen(true)}
      >
        Menu
      </button>
      {isOpen ? <div data-testid="menu-over-drawer" /> : null}
    </>
  );
}

function renderDrawer(
  section: (typeof CONVERSATION_OVERVIEW_DRAWER_SECTION)[keyof typeof CONVERSATION_OVERVIEW_DRAWER_SECTION],
) {
  return render(
    <ConversationOverviewDrawerProvider>
      <OpenSection section={section} />
      <MenuOverDrawer />
      <ConversationOverviewDrawerContent />
    </ConversationOverviewDrawerProvider>,
    {
      wrapper: ({ children }) => (
        <QueryClientProvider
          client={
            new QueryClient({
              defaultOptions: { queries: { retry: false } },
            })
          }
        >
          {children}
        </QueryClientProvider>
      ),
    },
  );
}

const getConversationPullRequestsSpy = vi.spyOn(
  GitProviderPullRequestStatusService,
  "getConversationPullRequests",
);

beforeEach(() => {
  getConversationPullRequestsSpy.mockReset();
  getConversationPullRequestsSpy.mockResolvedValue({
    status: "ok",
    source: "none",
    omittedCount: 0,
    pullRequests: [],
  });
});

function pullRequest(
  overrides: Partial<ConversationPullRequest>,
): ConversationPullRequest {
  return {
    repository: "openhands/agent-canvas",
    number: 10,
    title: "Fix overview",
    url: "https://github.com/openhands/agent-canvas/pull/10",
    lifecycle: "open",
    checks: "passing",
    review: "none",
    mergeable: "clean",
    updatedAt: null,
    unavailableReason: null,
    ...overrides,
  };
}

async function openPullRequests() {
  renderDrawer(CONVERSATION_OVERVIEW_DRAWER_SECTION.pull_requests);
  await userEvent.setup().click(screen.getByTestId("open-drawer-section"));
}

describe("ConversationOverviewDrawerContent pull requests", () => {
  it("shows each PR's repository, lifecycle, checks, review and conflicts as text", async () => {
    // Arrange
    getConversationPullRequestsSpy.mockResolvedValue({
      status: "ok",
      source: "tracked",
      omittedCount: 0,
      pullRequests: [
        pullRequest({
          checks: "failing",
          review: "approved",
          mergeable: "conflicting",
        }),
        pullRequest({
          number: 11,
          url: "https://github.com/openhands/agent-canvas/pull/11",
          lifecycle: "merged",
          checks: "none",
          mergeable: "not_applicable",
        }),
      ],
    });

    // Act
    await openPullRequests();

    // Assert
    const open = await screen.findByTestId("conversation-pull-request-10");
    expect(within(open).getByRole("link")).toHaveAttribute(
      "href",
      "https://github.com/openhands/agent-canvas/pull/10",
    );
    expect(open).toHaveTextContent("openhands/agent-canvas");
    for (const label of [
      "CONVERSATION$PR_LIFECYCLE_OPEN",
      "CONVERSATION$PR_CHECKS_FAILING",
      "CONVERSATION$PR_REVIEW_APPROVED",
      "CONVERSATION$PR_MERGE_CONFLICTING",
    ]) {
      expect(within(open).getByText(label)).toBeInTheDocument();
    }
    // A merged PR is not described as conflicting or clean.
    const merged = screen.getByTestId("conversation-pull-request-11");
    expect(
      within(merged).getByText("CONVERSATION$PR_LIFECYCLE_MERGED"),
    ).toBeInTheDocument();
    expect(merged).not.toHaveTextContent("CONVERSATION$PR_MERGE_");
  });

  it("explains a PR it could not read instead of guessing its state", async () => {
    // Arrange
    getConversationPullRequestsSpy.mockResolvedValue({
      status: "ok",
      source: "tracked",
      omittedCount: 0,
      pullRequests: [
        pullRequest({
          title: null,
          lifecycle: "unknown",
          checks: "unknown",
          review: "unknown",
          mergeable: "unknown",
          unavailableReason: "not_found",
        }),
      ],
    });

    // Act
    await openPullRequests();

    // Assert
    const row = await screen.findByTestId("conversation-pull-request-10");
    expect(row).toHaveTextContent("CONVERSATION$PR_UNAVAILABLE_NOT_FOUND");
    expect(row).not.toHaveTextContent("CONVERSATION$PR_CHECKS_");
  });

  it("says when PRs were matched by branch and how many were left out", async () => {
    // Arrange
    getConversationPullRequestsSpy.mockResolvedValue({
      status: "ok",
      source: "branch",
      omittedCount: 2,
      pullRequests: [pullRequest({})],
    });

    // Act
    await openPullRequests();

    // Assert
    expect(
      await screen.findByText("CONVERSATION$PR_FROM_BRANCH"),
    ).toBeInTheDocument();
    expect(screen.getByText("CONVERSATION$PR_OMITTED")).toBeInTheDocument();
  });

  it.each([
    [
      { status: "unsupported_provider" as const },
      "CONVERSATION$PR_STATUS_UNSUPPORTED",
    ],
    [
      { status: "unavailable" as const, reason: "auth_failed" as const },
      "CONVERSATION$PR_UNAVAILABLE_AUTH_FAILED",
    ],
    [
      {
        status: "ok" as const,
        source: "none" as const,
        omittedCount: 0,
        pullRequests: [],
      },
      "CONVERSATION$PR_NONE_FOR_CONVERSATION",
    ],
  ])(
    "shows an explicit state instead of an empty list (%o)",
    async (result, label) => {
      // Arrange
      getConversationPullRequestsSpy.mockResolvedValue(result);

      // Act
      await openPullRequests();

      // Assert
      expect(await screen.findByText(label)).toBeInTheDocument();
    },
  );
});

describe("ConversationOverviewDrawerContent", () => {
  it("places the close button left of the title", async () => {
    const user = userEvent.setup();
    renderDrawer(CONVERSATION_OVERVIEW_DRAWER_SECTION.pull_requests);

    await user.click(screen.getByTestId("open-drawer-section"));

    const header = screen
      .getByTestId("conversation-overview-drawer-content")
      .querySelector("header");
    expect(header).not.toBeNull();
    expect(header).toHaveClass("h-10");
    expect(header).toHaveClass("min-h-10");
    expect(header).toHaveClass("pr-4");
    const closeButton = within(header as HTMLElement).getByTestId(
      "conversation-overview-drawer-close",
    );
    const title = within(header as HTMLElement).getByRole("heading");
    expect(
      closeButton.compareDocumentPosition(title) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  it("lets a menu opened over the drawer take the first Escape", async () => {
    const user = userEvent.setup();
    renderDrawer(CONVERSATION_OVERVIEW_DRAWER_SECTION.pull_requests);

    await user.click(screen.getByTestId("open-drawer-section"));
    await user.click(screen.getByTestId("open-menu-over-drawer"));

    await user.keyboard("{Escape}");
    expect(screen.queryByTestId("menu-over-drawer")).not.toBeInTheDocument();
    expect(
      screen.getByTestId("conversation-overview-drawer-content"),
    ).toBeInTheDocument();

    await user.keyboard("{Escape}");
    expect(
      screen.queryByTestId("conversation-overview-drawer-content"),
    ).not.toBeInTheDocument();
  });

  it("places the view-on-provider link in the header for pull requests", async () => {
    const user = userEvent.setup();
    renderDrawer(CONVERSATION_OVERVIEW_DRAWER_SECTION.pull_requests);

    await user.click(screen.getByTestId("open-drawer-section"));

    const header = screen
      .getByTestId("conversation-overview-drawer-content")
      .querySelector("header");
    const externalLink = within(header as HTMLElement).getByTestId(
      "conversation-overview-pull_requests-open-external",
    );

    expect(externalLink).toHaveAttribute(
      "href",
      "https://github.com/openhands/agent-canvas/pulls",
    );
    expect(externalLink).toHaveTextContent(
      "CONVERSATION$OVERVIEW_VIEW_ON_PROVIDER",
    );
    expect(
      screen
        .getByTestId("conversation-overview-pull_requests-panel")
        .querySelector(
          '[data-testid="conversation-overview-pull_requests-open-external"]',
        ),
    ).toBeNull();
  });
});

function DrawerShell({
  isMobile = false,
  resizable = true,
}: {
  isMobile?: boolean;
  resizable?: boolean;
}) {
  const containerRef = React.useRef<HTMLDivElement>(null);
  return (
    <div ref={containerRef} data-testid="drawer-resize-container">
      <OpenSection
        section={CONVERSATION_OVERVIEW_DRAWER_SECTION.pull_requests}
      />
      <ConversationOverviewDrawer
        isMobile={isMobile}
        resizeContainerRef={resizable ? containerRef : undefined}
      />
    </div>
  );
}

function renderShell(props: React.ComponentProps<typeof DrawerShell> = {}) {
  return render(
    <QueryClientProvider
      client={
        new QueryClient({ defaultOptions: { queries: { retry: false } } })
      }
    >
      <ConversationOverviewDrawerProvider>
        <DrawerShell {...props} />
      </ConversationOverviewDrawerProvider>
    </QueryClientProvider>,
  );
}

// Evaluate production declarations against the real DOM, including nested state
// selectors. This checks the CSS reader/writer contract rather than class names.
async function compiledStyles(container: HTMLElement) {
  const compiler = await compile(
    await readFile(resolve("src/tailwind.css"), "utf8"),
    { base: resolve("src"), onDependency: () => {} },
  );
  const css = compiler.build(
    [...container.querySelectorAll<HTMLElement>("[class]")].flatMap(
      (element) => [...element.classList],
    ),
  );
  const root = postcss.parse(css);
  return (element: HTMLElement) => {
    const declarations = new Map<string, string>();
    root.walkRules((rule) => {
      let selector = rule.selector;
      for (
        let parent: Container | Document | undefined = rule.parent;
        parent;
        parent = parent.parent
      ) {
        if (parent.type === "rule") {
          selector = selector.replaceAll("&", (parent as Rule).selector);
        }
      }
      try {
        if (!element.matches(selector)) return;
      } catch {
        return; // Browser-specific pseudo-elements do not select this element.
      }
      rule.nodes.forEach((node) => {
        if (node.type === "decl") declarations.set(node.prop, node.value);
      });
    });
    const width = declarations.get("width");
    const widthProperty = /^var\((--[\w-]+)\)$/.exec(width ?? "")?.[1];
    let widthValue = width;
    if (widthProperty) {
      let owner: HTMLElement | null = element;
      while (owner && !owner.style.getPropertyValue(widthProperty)) {
        owner = owner.parentElement;
      }
      widthValue = owner?.style.getPropertyValue(widthProperty);
    }
    return { declarations, widthValue, widthProperty, css };
  };
}

describe("ConversationOverviewDrawer desktop CSS", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("keeps the inner pixel width while the outer drawer opens and closes", async () => {
    // Arrange: real section context and the production desktop shell.
    const user = userEvent.setup();
    const view = renderShell();
    const outer = screen.getByTestId("conversation-overview-drawer");
    const inner = outer.firstElementChild as HTMLElement;
    await user.click(screen.getByTestId("open-drawer-section"));
    const styles = await compiledStyles(view.container);

    // Act + Assert: CSS, not direct styles, owns width and motion properties.
    expect(outer.style.width).toBe("");
    expect(inner.style.width).toBe("");
    expect(outer.style.transitionProperty).toBe("");
    expect(styles(outer).widthProperty).toBeDefined();
    expect(styles(outer).widthValue).toBe("420px");
    expect(styles(inner).widthValue).toBe("420px");
    expect(styles(outer).declarations.get("transition-property")).toBe(
      "width, transform, opacity",
    );
    expect(styles(outer).declarations.get("transition-duration")).toBe("300ms");
    expect(styles(outer).declarations.get("transition-timing-function")).toBe(
      "var(--ease-in-out)",
    );

    await user.click(screen.getByTestId("conversation-overview-drawer-close"));
    expect(outer).toHaveAttribute("aria-hidden", "true");
    expect(styles(outer).widthValue).toBe("0px");
    expect(styles(inner).widthValue).toBe("420px");
    expect(
      screen.queryByTestId("conversation-overview-drawer-content"),
    ).toBeNull();
    await user.click(screen.getByTestId("open-drawer-section"));
    expect(styles(outer).widthValue).toBe("420px");
  });

  it("follows real drag bounds, excludes width transitions, and restores saved width", async () => {
    // Arrange: right-anchored pointer coordinates are measured by the real hook.
    const user = userEvent.setup();
    const view = renderShell();
    await user.click(screen.getByTestId("open-drawer-section"));
    vi.spyOn(
      screen.getByTestId("drawer-resize-container"),
      "getBoundingClientRect",
    ).mockReturnValue(DOMRect.fromRect({ x: 100, width: 1000, height: 600 }));
    const outer = screen.getByTestId("conversation-overview-drawer");
    const inner = outer.firstElementChild as HTMLElement;
    const styles = await compiledStyles(view.container);
    const handle = screen.getByTestId(
      "conversation-overview-drawer-resize-handle",
    );

    // Act: keep the drag active while checking the CSS state and both clamps.
    fireEvent.mouseDown(handle.querySelector(".cursor-ew-resize")!);
    expect(document.querySelector("[data-panel-drag-shield]")).not.toBeNull();
    expect(styles(outer).declarations.get("transition-property")).toBe(
      "transform, opacity",
    );
    for (const [clientX, expectedWidth] of [
      [600, "500px"],
      [0, "640px"],
      [1200, "300px"],
      [600, "500px"],
    ] as const) {
      fireEvent.mouseMove(document, { clientX });
      expect(styles(outer).widthValue).toBe(expectedWidth);
      expect(styles(inner).widthValue).toBe(expectedWidth);
    }
    fireEvent.mouseUp(document);

    // Assert: releasing the drag restores motion and persists the final value.
    expect(styles(outer).declarations.get("transition-property")).toBe(
      "width, transform, opacity",
    );
    expect(document.querySelector("[data-panel-drag-shield]")).toBeNull();
    expect(document.body.style.cursor).toBe("");
    expect(document.body.style.userSelect).toBe("");
    expect(
      JSON.parse(
        localStorage.getItem(CONVERSATION_OVERVIEW_DRAWER_WIDTH_STORAGE_KEY)!,
      ),
    ).toBe(500);
    await user.click(screen.getByTestId("conversation-overview-drawer-close"));
    expect(styles(outer).widthValue).toBe("0px");
    expect(styles(inner).widthValue).toBe("500px");
    await user.click(screen.getByTestId("open-drawer-section"));
    expect(styles(outer).widthValue).toBe("500px");
    view.unmount();
    const restored = renderShell();
    await user.click(screen.getByTestId("open-drawer-section"));
    expect(
      (await compiledStyles(restored.container))(
        screen.getByTestId("conversation-overview-drawer"),
      ).widthValue,
    ).toBe("500px");
  });

  it("cleans an in-progress resize when the drawer unmounts", async () => {
    const user = userEvent.setup();
    const view = renderShell();
    await user.click(screen.getByTestId("open-drawer-section"));
    const handle = screen.getByTestId(
      "conversation-overview-drawer-resize-handle",
    );
    const savedWidth = localStorage.getItem(
      CONVERSATION_OVERVIEW_DRAWER_WIDTH_STORAGE_KEY,
    );
    fireEvent.mouseDown(handle.querySelector(".cursor-ew-resize")!);
    expect(document.querySelector("[data-panel-drag-shield]")).not.toBeNull();

    view.unmount();
    fireEvent.mouseMove(document, { clientX: 0 });
    fireEvent.mouseUp(document);

    expect(document.querySelector("[data-panel-drag-shield]")).toBeNull();
    expect(document.body.style.cursor).toBe("");
    expect(document.body.style.userSelect).toBe("");
    expect(
      localStorage.getItem(CONVERSATION_OVERVIEW_DRAWER_WIDTH_STORAGE_KEY),
    ).toBe(savedWidth);
  });

  it("retains the phone content branch and ref-less desktop boundary", async () => {
    const user = userEvent.setup();
    const phone = renderShell({ isMobile: true });
    await user.click(screen.getByTestId("open-drawer-section"));
    const content = screen.getByTestId("conversation-overview-drawer-content");
    const styles = await compiledStyles(phone.container);
    expect(styles(content).declarations.get("max-height")).toBe(
      "min(50vh, 420px)",
    );
    expect(screen.queryByTestId("conversation-overview-drawer")).toBeNull();
    expect(
      screen.queryByTestId("conversation-overview-drawer-resize-handle"),
    ).toBeNull();
    phone.unmount();

    renderShell({ resizable: false });
    await user.click(screen.getByTestId("open-drawer-section"));
    expect(
      screen.getByTestId("conversation-overview-drawer-content"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("conversation-overview-drawer-resize-handle"),
    ).toBeNull();
  });
});
