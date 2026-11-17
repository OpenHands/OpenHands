import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  MemoryRouter,
  Route,
  Routes,
  useLocation,
  useNavigate,
} from "react-router";

import AutomationService from "#/api/automation-service/automation-service.api";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import type { LicenseStatus } from "#/api/cloud/license-service.api";
import { callCloudProxy } from "#/api/cloud/proxy";
import type { Worklog, WorklogEntry } from "#/api/cloud/worklog-service.api";
import { NavigationProvider } from "#/context/navigation-context";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { WORKLOG_POLL_INTERVAL_MS } from "#/hooks/query/use-worklogs";
import FactoryDetail from "#/routes/factory-detail";
import FactoryWorklogs from "#/routes/factory-worklogs";
import type { Factory } from "#/types/automation";

vi.mock("#/manifests/manifest-sources", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("#/manifests/manifest-sources")>();
  const { createInterfaceManifestWithSubPages } =
    await import("../manifests/manifest-test-data");
  return {
    ...actual,
    AUTOMATION_INTERFACE_CANDIDATE: createInterfaceManifestWithSubPages(),
  };
});

vi.mock("#/api/automation-service/automation-service.api", () => ({
  default: {
    getCapabilities: vi.fn(),
    supportsFactories: vi.fn(),
    getFactory: vi.fn(),
    getAutomations: vi.fn(),
    getAutomationRuns: vi.fn(),
    listAutomationTags: vi.fn(),
    listServerDrafts: vi.fn(),
    supportsAutomationDrafts: vi.fn(),
    checkHealth: vi.fn(),
  },
}));

vi.mock("#/api/cloud/proxy", () => ({ callCloudProxy: vi.fn() }));

const CLOUD_HOST = "https://ohe.example.com";
const ORG_ID = "org-1";
const FACTORY_ID = "f-core";
const WORKLOGS_URL = `/api/organizations/${ORG_ID}/factories/${FACTORY_ID}/worklogs`;

const cloudBackend: Backend = {
  id: "cloud-1",
  name: "OpenHands Enterprise",
  host: CLOUD_HOST,
  apiKey: "token",
  kind: "cloud",
};

const coreFactory: Factory = {
  id: FACTORY_ID,
  name: "Core SDLC",
  automation_count: 2,
  created_at: "2026-11-01T00:00:00Z",
  updated_at: "2026-11-01T00:00:00Z",
};

/** An in-memory stand-in for the Enterprise worklog routes. */
interface FakeWorklog {
  worklog: Omit<Worklog, "entry_count" | "json_entry_count" | "last_entry_at">;
  entries: WorklogEntry[];
}

let worklogsById: Record<string, FakeWorklog>;
let nextEntryId: number;
let licenseFeatures: LicenseStatus["features"];
let worklogsStatus: number | null;

function makeWorklog(slug: string, name: string, isDefault = false) {
  return {
    id: `wl-${slug}`,
    factory_id: FACTORY_ID,
    slug,
    name,
    is_default: isDefault,
    created_at: "2026-11-01T00:00:00Z",
    updated_at: "2026-11-01T00:00:00Z",
  };
}

function addEntry(
  slug: string,
  entry: Partial<WorklogEntry> & Pick<WorklogEntry, "body">,
) {
  const id = nextEntryId;
  nextEntryId += 1;
  worklogsById[slug].entries.push({
    id,
    created_at: new Date(Date.UTC(2026, 10, 10, 0, 0, id)).toISOString(),
    content_type: "markdown",
    title: null,
    author: { kind: "automation", id: "a-1", name: "Orchestrator" },
    run_id: null,
    ...entry,
  });
}

function summarize({ worklog, entries }: FakeWorklog): Worklog {
  return {
    ...worklog,
    entry_count: entries.length,
    json_entry_count: entries.filter((e) => e.content_type === "json").length,
    last_entry_at: entries.at(-1)?.created_at ?? null,
  };
}

function fakeEnterprise({ path }: { path: string }): unknown {
  const [pathname, search = ""] = path.split("?");
  const params = new URLSearchParams(search);

  if (pathname === "/api/license/status") {
    return {
      state: "valid",
      reason: null,
      offline: false,
      type: "paid",
      license_ends_at: null,
      expires_at: null,
      last_checked_at: null,
      customer_center_url: "https://customercenter.openhands.dev",
      features: licenseFeatures,
    } satisfies LicenseStatus;
  }
  if (pathname === "/api/organizations") {
    return { items: [{ id: ORG_ID, name: "Org" }], current_org_id: ORG_ID };
  }
  if (pathname.startsWith(`/api/organizations/${ORG_ID}/members/`)) {
    return { org_id: ORG_ID, user_id: "u-1", email: "ada@example.com" };
  }
  if (pathname.startsWith(WORKLOGS_URL)) {
    if (worklogsStatus !== null) {
      throw Object.assign(new Error("Worklogs unavailable"), {
        status: worklogsStatus,
      });
    }
    if (pathname === WORKLOGS_URL) {
      return { items: Object.values(worklogsById).map(summarize) };
    }
    const slug = pathname.slice(WORKLOGS_URL.length + 1).split("/")[0];
    const contentType = params.get("content_type");
    const cursor = params.get("since");
    const limit = Number(params.get("limit"));
    expect(params.get("order")).toBe("desc");
    const matching = worklogsById[slug].entries
      .filter((e) => !contentType || e.content_type === contentType)
      .filter((e) => cursor === null || e.id < Number(cursor))
      .sort((a, b) => b.id - a.id);
    const items = matching.slice(0, limit);
    return {
      items,
      next_cursor: items.length ? String(items.at(-1)!.id) : null,
      has_more: matching.length > limit,
    };
  }
  throw new Error(`Unexpected Enterprise request: ${path}`);
}

function RouterNavigation({ children }: { children: React.ReactNode }) {
  const location = useLocation();
  const navigate = useNavigate();
  return (
    <NavigationProvider
      value={{
        currentPath: location.pathname,
        conversationId: null,
        isNavigating: false,
        navigate: (to, options) => navigate(to, { replace: options?.replace }),
      }}
    >
      {children}
    </NavigationProvider>
  );
}

function renderAt(path: string) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <ActiveBackendProvider>
        <MemoryRouter initialEntries={[path]}>
          <RouterNavigation>
            <Routes>
              <Route path="/factories/:factoryId" element={<FactoryDetail />} />
              <Route
                path="/factories/:factoryId/worklogs"
                element={<FactoryWorklogs />}
              />
            </Routes>
          </RouterNavigation>
        </MemoryRouter>
      </ActiveBackendProvider>
    </QueryClientProvider>,
  );
}

const WORKLOGS_PATH = `/factories/${FACTORY_ID}/worklogs`;

function entryBodies() {
  return screen
    .getAllByTestId("worklog-entry")
    .map((entry) => entry.textContent ?? "");
}

beforeEach(() => {
  __resetActiveStoreForTests();
  vi.stubEnv("VITE_LOCK_TO_CLOUD", CLOUD_HOST);
  worklogsById = {
    default: { worklog: makeWorklog("default", "Default", true), entries: [] },
  };
  nextEntryId = 1;
  licenseFeatures = { sso: false, worklogs: true };
  worklogsStatus = null;
  vi.mocked(callCloudProxy).mockImplementation(async (req) =>
    fakeEnterprise(req),
  );
  vi.mocked(AutomationService.getCapabilities).mockResolvedValue({
    ready: true,
    features: ["factories"],
    triggerKinds: ["cron"],
    eventSources: [],
    eventTypes: [],
    triggers: {},
  });
  vi.mocked(AutomationService.supportsFactories).mockReturnValue(true);
  vi.mocked(AutomationService.supportsAutomationDrafts).mockReturnValue(false);
  vi.mocked(AutomationService.getFactory).mockResolvedValue(coreFactory);
  setRegisteredBackends([cloudBackend]);
  setActiveSelection({ backendId: cloudBackend.id, orgId: ORG_ID });
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllEnvs();
  vi.clearAllMocks();
  __resetActiveStoreForTests();
  window.localStorage.clear();
});

describe("Factory worklogs", () => {
  it("adds a Worklogs item to the factory navigation when the license grants worklogs", async () => {
    // Arrange + Act
    renderAt(`/factories/${FACTORY_ID}`);

    // Assert
    const links = await screen.findAllByTestId("factory-navigation-worklogs");
    expect(links[0]).toHaveAttribute("href", WORKLOGS_PATH);
  });

  it("leaves the Worklogs item out and explains why on a deep link when the license lacks worklogs", async () => {
    // Arrange
    licenseFeatures = { sso: false };

    // Act
    renderAt(WORKLOGS_PATH);

    // Assert
    expect(await screen.findByTestId("worklogs-unsupported")).toBeVisible();
    expect(
      screen.queryByTestId("factory-navigation-worklogs"),
    ).not.toBeInTheDocument();
    expect(
      screen.getAllByTestId("factory-navigation-automations").length,
    ).toBeGreaterThan(0);
  });

  it("shows entries newest first and renders their GitHub-flavored markdown", async () => {
    // Arrange
    addEntry("default", { body: "oldest note" });
    addEntry("default", {
      body: "## Launched\n\n| run | result |\n| --- | --- |\n| 7 | ok |\n\n- [x] shipped",
    });

    // Act
    renderAt(WORKLOGS_PATH);

    // Assert
    const entries = await screen.findAllByTestId("worklog-entry");
    expect(entries).toHaveLength(2);
    expect(within(entries[0]).getByRole("heading", { name: "Launched" }));
    expect(within(entries[0]).getByRole("table")).toBeVisible();
    expect(within(entries[0]).getByRole("checkbox")).toBeChecked();
    expect(entries[1]).toHaveTextContent("oldest note");
    expect(entries[0]).toHaveTextContent("Orchestrator");
  });

  it("names a person who wrote an entry by their member email", async () => {
    // Arrange
    addEntry("default", {
      body: "INSTRUCTION: pause",
      author: { kind: "user", id: "u-1", name: null },
    });

    // Act
    renderAt(WORKLOGS_PATH);

    // Assert
    const author = await screen.findByTestId("worklog-entry-author");
    await waitFor(() => expect(author).toHaveTextContent("ada@example.com"));
  });

  it("shows no tabs for a single worklog", async () => {
    // Arrange
    addEntry("default", { body: "only log" });

    // Act
    renderAt(WORKLOGS_PATH);

    // Assert
    await screen.findByTestId("worklog-entry");
    expect(screen.queryByTestId("worklog-tabs")).not.toBeInTheDocument();
  });

  it("shows a tab per worklog, default first, and switches between them", async () => {
    // Arrange
    const user = userEvent.setup();
    worklogsById.release = {
      worklog: makeWorklog("release", "Release"),
      entries: [],
    };
    addEntry("default", { body: "default entry" });
    addEntry("release", { body: "release entry" });

    // Act
    renderAt(WORKLOGS_PATH);

    // Assert
    const tabs = await screen.findAllByRole("tab");
    expect(tabs.map((tab) => tab.textContent)).toEqual(["Default", "Release"]);
    expect(tabs[0]).toHaveAttribute("aria-selected", "true");
    expect(await screen.findByText("default entry")).toBeVisible();

    await user.click(tabs[1]);

    expect(await screen.findByText("release entry")).toBeVisible();
    expect(screen.queryByText("default entry")).not.toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Release" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
  });

  it("opens the worklog named in the URL", async () => {
    // Arrange
    worklogsById.release = {
      worklog: makeWorklog("release", "Release"),
      entries: [],
    };
    addEntry("release", { body: "release entry" });

    // Act
    renderAt(`${WORKLOGS_PATH}?worklog=release`);

    // Assert
    expect(await screen.findByText("release entry")).toBeVisible();
  });

  it("pages through older entries on request", async () => {
    // Arrange
    const user = userEvent.setup();
    for (let n = 1; n <= 25; n += 1) addEntry("default", { body: `note ${n}` });

    // Act
    renderAt(WORKLOGS_PATH);

    // Assert
    expect(await screen.findAllByTestId("worklog-entry")).toHaveLength(20);
    expect(entryBodies()[0]).toContain("note 25");
    expect(entryBodies()[19]).toContain("note 6");

    await user.click(screen.getByTestId("worklog-load-more"));

    await waitFor(() =>
      expect(screen.getAllByTestId("worklog-entry")).toHaveLength(25),
    );
    expect(entryBodies()[24]).toContain("note 1");
    expect(screen.queryByTestId("worklog-load-more")).not.toBeInTheDocument();
  });

  it("hides JSON entries until asked and says how many are hidden", async () => {
    // Arrange
    const user = userEvent.setup();
    addEntry("default", { body: "a note" });
    addEntry("default", {
      body: '{"launched":3}',
      content_type: "json",
      title: "Launch summary",
    });

    // Act
    renderAt(WORKLOGS_PATH);

    // Assert
    await screen.findByText("a note");
    expect(screen.queryByTestId("worklog-entry-json")).not.toBeInTheDocument();

    await user.click(screen.getByTestId("worklog-json-toggle"));

    expect(await screen.findByTestId("worklog-entry-json")).toHaveTextContent(
      '{"launched":3}',
    );
    expect(screen.getByText("Launch summary")).toBeVisible();
  });

  it("adds an entry written while the page is open to the top of the list", async () => {
    // Arrange
    vi.useFakeTimers({ shouldAdvanceTime: true });
    addEntry("default", { body: "earlier note" });
    renderAt(WORKLOGS_PATH);
    await screen.findByText("earlier note");

    // Act
    addEntry("default", { body: "fresh note" });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(WORKLOG_POLL_INTERVAL_MS + 100);
    });

    // Assert
    await waitFor(() => expect(entryBodies()[0]).toContain("fresh note"));
    expect(entryBodies()[1]).toContain("earlier note");
  });

  it.each([403, 503])(
    "says worklogs are unavailable when the service answers %i",
    async (status) => {
      // Arrange
      worklogsStatus = status;

      // Act
      renderAt(WORKLOGS_PATH);

      // Assert
      expect(await screen.findByTestId("worklogs-unavailable")).toBeVisible();
    },
  );
});
