import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import React from "react";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router";

import AutomationService from "#/api/automation-service/automation-service.api";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import AutomationsList from "#/routes/automations-list";
import { I18nKey } from "#/i18n/declaration";
import type { Backend } from "#/api/backend-registry/types";
import type { Automation } from "#/types/automation";

// A manifest that declares the tag filter and the tags endpoint, as the
// extensions package does once it ships them.
vi.mock("#/manifests/manifest-sources", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("#/manifests/manifest-sources")>();
  const { createInterfaceManifestWithSubPages } =
    await import("../manifests/manifest-test-data");
  const manifest = createInterfaceManifestWithSubPages();
  return {
    ...actual,
    AUTOMATION_INTERFACE_CANDIDATE: {
      ...manifest,
      endpoints: { ...manifest.endpoints, tags: "/v1/tags" },
      pages: {
        ...manifest.pages,
        list: {
          ...manifest.pages.list,
          filters: [
            ...manifest.pages.list.filters!,
            { id: "tag", label: "Filter widgets by tag", allLabel: "Any tag" },
          ],
        },
      },
    },
  };
});

vi.mock("#/api/automation-service/automation-service.api", () => ({
  default: {
    getAutomations: vi.fn(),
    listAutomationTags: vi.fn(),
    getAutomationRuns: vi.fn(),
    listServerDrafts: vi.fn(),
    getCapabilities: vi.fn(),
    supportsAutomationDrafts: vi.fn(),
    checkHealth: vi.fn(),
  },
}));

const localBackend: Backend = {
  id: "local-1",
  name: "Local 1",
  host: "http://localhost:8000",
  apiKey: "session-key",
  kind: "local",
};

function createAutomation(overrides: Partial<Automation>): Automation {
  return {
    id: "a-1",
    name: "Alpha widget",
    trigger: { type: "cron", schedule: "0 9 * * *" },
    enabled: true,
    prompt: "Watch the widgets",
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    tags: [],
    ...overrides,
  };
}

const factoryAutomation = createAutomation({
  id: "a-factory",
  name: "PR reviewer",
  tags: ["software-factory"],
});
const otherAutomation = createAutomation({ id: "a-other", name: "Digest" });
const reviewAutomation = createAutomation({
  id: "a-review",
  name: "Code reviewer",
  tags: ["software-factory", "code-review"],
});

function renderList() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <ActiveBackendProvider>
        <MemoryRouter initialEntries={["/automations"]}>
          <AutomationsList />
        </MemoryRouter>
      </ActiveBackendProvider>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  __resetActiveStoreForTests();
  vi.mocked(AutomationService.checkHealth).mockResolvedValue({ status: "ok" });
  vi.mocked(AutomationService.listServerDrafts).mockResolvedValue({
    drafts: [],
    total: 0,
  });
  vi.mocked(AutomationService.getCapabilities).mockResolvedValue({
    ready: true,
    features: [],
    triggerKinds: ["cron"],
    eventSources: [],
    eventTypes: [],
    triggers: {},
  });
  vi.mocked(AutomationService.supportsAutomationDrafts).mockReturnValue(false);
  vi.mocked(AutomationService.getAutomationRuns).mockResolvedValue({
    runs: [],
    total: 0,
  });
  vi.mocked(AutomationService.listAutomationTags).mockResolvedValue([
    { tag: "software-factory", count: 1 },
  ]);
  vi.mocked(AutomationService.getAutomations).mockImplementation(
    async (_limit, _offset, _createdBy, tag) => {
      const automations = tag
        ? [factoryAutomation]
        : [factoryAutomation, otherAutomation];
      return { automations, total: automations.length };
    },
  );
  setRegisteredBackends([localBackend]);
  setActiveSelection({ backendId: localBackend.id });
});

afterEach(() => {
  vi.clearAllMocks();
  __resetActiveStoreForTests();
  window.localStorage.clear();
});

describe("AutomationsList — tag filter", () => {
  it("asks the server for the selected tag's automations only", async () => {
    // Arrange
    const user = userEvent.setup();
    renderList();
    await screen.findByTestId("automation-card-a-other");

    // Act
    await user.click(
      within(screen.getByTestId("automations-filters")).getByTestId(
        "dropdown-trigger",
      ),
    );
    const field = await screen.findByTestId("automations-filter-tag");
    await user.click(within(field).getByTestId("dropdown-trigger"));
    await user.click(
      screen.getByTestId("automations-filter-tag-software-factory"),
    );

    // Assert
    await waitFor(() =>
      expect(
        screen.queryByTestId("automation-card-a-other"),
      ).not.toBeInTheDocument(),
    );
    expect(screen.getByTestId("automation-card-a-factory")).toBeInTheDocument();
    expect(AutomationService.getAutomations).toHaveBeenLastCalledWith(
      50,
      0,
      undefined,
      "software-factory",
    );
  });
});

describe("AutomationsList — group by tag", () => {
  const sectionTitles = () =>
    screen
      .getAllByRole("heading", { level: 2 })
      .map((heading) => heading.textContent);

  const sectionOf = (title: string) =>
    within(
      screen
        .getByRole("heading", { level: 2, name: title })
        .closest("section") as HTMLElement,
    );

  const groupByTag = async (user: ReturnType<typeof userEvent.setup>) => {
    await user.click(
      within(screen.getByTestId("automations-group-by")).getByTestId(
        "dropdown-trigger",
      ),
    );
    await user.click(screen.getByTestId("automations-group-by-tag"));
  };

  beforeEach(() => {
    // The service lists the most-used tag first, which is not alphabetical.
    vi.mocked(AutomationService.listAutomationTags).mockResolvedValue([
      { tag: "software-factory", count: 2 },
      { tag: "code-review", count: 1 },
    ]);
    vi.mocked(AutomationService.getAutomations).mockResolvedValue({
      automations: [factoryAutomation, reviewAutomation, otherAutomation],
      total: 3,
    });
  });

  it("lists a section per tag in alphabetical order, then the untagged automations", async () => {
    // Arrange
    const user = userEvent.setup();
    renderList();
    await screen.findByTestId("automation-card-a-other");

    // Act
    await groupByTag(user);

    // Assert
    expect(sectionTitles()).toEqual([
      "code-review",
      "software-factory",
      I18nKey.AUTOMATIONS$NO_TAGS,
    ]);
    expect(
      sectionOf(I18nKey.AUTOMATIONS$NO_TAGS).getByTestId(
        "automation-card-a-other",
      ),
    ).toBeInTheDocument();
  });

  it("lists an automation under each of its tags", async () => {
    // Arrange
    const user = userEvent.setup();
    renderList();
    await screen.findByTestId("automation-card-a-other");

    // Act
    await groupByTag(user);

    // Assert
    expect(
      sectionOf("code-review").getByTestId("automation-card-a-review"),
    ).toBeInTheDocument();
    expect(
      sectionOf("software-factory").getByTestId("automation-card-a-review"),
    ).toBeInTheDocument();
  });

  it("groups only the automations that match the search", async () => {
    // Arrange
    const user = userEvent.setup();
    renderList();
    await screen.findByTestId("automation-card-a-other");
    await groupByTag(user);

    // Act
    await user.type(
      screen.getByLabelText(I18nKey.AUTOMATIONS$SEARCH_PLACEHOLDER),
      reviewAutomation.name,
    );

    // Assert
    expect(sectionTitles()).toEqual(["code-review", "software-factory"]);
    expect(
      screen.queryByTestId("automation-card-a-factory"),
    ).not.toBeInTheDocument();
  });

  it("keeps the grouping for the next visit", async () => {
    // Arrange
    const user = userEvent.setup();
    const { unmount } = renderList();
    await screen.findByTestId("automation-card-a-other");
    await groupByTag(user);
    unmount();

    // Act
    renderList();

    // Assert
    expect(
      await screen.findByRole("heading", { level: 2, name: "code-review" }),
    ).toBeInTheDocument();
  });

  it("returns to the status sections when the org has no tags left", async () => {
    // Arrange
    const user = userEvent.setup();
    const { unmount } = renderList();
    await screen.findByTestId("automation-card-a-other");
    await groupByTag(user);
    unmount();
    vi.mocked(AutomationService.listAutomationTags).mockResolvedValue([]);
    vi.mocked(AutomationService.getAutomations).mockResolvedValue({
      automations: [otherAutomation],
      total: 1,
    });

    // Act
    renderList();
    await screen.findByTestId("automation-card-a-other");

    // Assert
    expect(sectionTitles()).toEqual([I18nKey.AUTOMATIONS$ACTIVE]);
    expect(
      screen.queryByTestId("automations-group-by"),
    ).not.toBeInTheDocument();
  });
});
