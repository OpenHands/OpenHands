import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router";
import { http, HttpResponse } from "msw";
import { server } from "#/mocks/node";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { resetAutomationMockData } from "#/mocks/automation-handlers";
import type { Backend } from "#/api/backend-registry/types";
import AutomationEvents, { clientLoader } from "#/routes/automation-events";

vi.mock("react-i18next", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react-i18next")>();
  const translations = await import("#/i18n/translation.json");
  return {
    ...actual,
    useTranslation: () => ({
      t: (key: keyof typeof translations.default) =>
        translations.default[key]?.en ?? key,
    }),
  };
});

vi.mock("#/manifests/manifest-sources", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("#/manifests/manifest-sources")>();
  const { createInterfaceManifestWithSubPages } =
    await import("../manifests/manifest-test-data");
  const base = createInterfaceManifestWithSubPages();
  return {
    ...actual,
    AUTOMATION_INTERFACE_CANDIDATE: {
      ...base,
      routes: { ...base.routes, events: "/automations/events" },
      pages: {
        ...base.pages,
        events: { title: "Event sources", description: "Manage webhooks" },
      },
      endpoints: { ...base.endpoints, webhooks: "/v1/webhooks" },
      navigation: {
        ...base.navigation,
        subPages: [
          base.navigation.subPages![0],
          { page: "events", label: "Events", icon: "activity" },
          base.navigation.subPages![1],
        ],
      },
    },
  };
});

const local: Backend = {
  kind: "local",
  id: "events-local",
  name: "Local",
  host: "https://local.example.test",
  apiKey: "test-session",
};
const cloud: Backend = {
  kind: "cloud",
  id: "events-cloud",
  name: "Cloud",
  host: "https://cloud.example.test",
  apiKey: "test-key",
};

function renderEvents() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  render(
    <QueryClientProvider client={client}>
      <ActiveBackendProvider>
        <MemoryRouter initialEntries={["/automations/events"]}>
          <AutomationEvents />
        </MemoryRouter>
      </ActiveBackendProvider>
    </QueryClientProvider>,
  );
  return client;
}

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  resetAutomationMockData();
  setRegisteredBackends([local, cloud]);
  setActiveSelection({ backendId: local.id });
});

afterEach(() => {
  resetAutomationMockData();
  __resetActiveStoreForTests();
});

describe("custom event sources", () => {
  it("mounts an admitted event route with ordered navigation and an informational empty warning", async () => {
    expect(clientLoader()).toBeNull();
    renderEvents();
    expect(
      await screen.findByText("No custom webhook event sources yet."),
    ).toBeInTheDocument();
    expect(screen.getByText(/publicly reachable/)).toHaveTextContent(
      /scheduled polling/,
    );
    const nav = within(screen.getByTestId("automations-navbar-desktop"));
    expect(nav.getAllByRole("link").map((link) => link.textContent)).toEqual([
      "Widget dashboard",
      "Events",
      "Widget templates",
    ]);
  });

  it("creates a source, shows a generated secret once, and keeps it out of React Query", async () => {
    const user = userEvent.setup();
    const client = renderEvents();
    await user.click(
      await screen.findByRole("button", { name: "Add event source" }),
    );
    await user.type(
      screen.getByRole("textbox", { name: "Name" }),
      "Deployments",
    );
    await user.type(
      screen.getByRole("textbox", { name: /Source identifier/ }),
      "deployments",
    );
    await user.click(screen.getByRole("button", { name: "Create" }));

    expect(
      await screen.findByText("mock-generated-signing-secret"),
    ).toBeInTheDocument();
    expect(
      JSON.stringify(
        client
          .getQueryCache()
          .getAll()
          .map((query) => query.state.data),
      ),
    ).not.toContain("mock-generated-signing-secret");
    expect(
      JSON.stringify(
        client
          .getMutationCache()
          .getAll()
          .map((mutation) => mutation.state),
      ),
    ).not.toContain("mock-generated-signing-secret");
    expect(await screen.findByText("Deployments")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Done" }));
    expect(
      screen.queryByText("mock-generated-signing-secret"),
    ).not.toBeInTheDocument();
  });

  it("never echoes a supplied secret, including on a duplicate-source failure", async () => {
    const user = userEvent.setup();
    const client = renderEvents();
    await user.click(
      await screen.findByRole("button", { name: "Add event source" }),
    );
    await user.type(
      screen.getByRole("textbox", { name: "Name" }),
      "Deployments",
    );
    await user.type(
      screen.getByRole("textbox", { name: /Source identifier/ }),
      "deployments",
    );
    await user.click(screen.getByText("advanced settings"));
    const supplied = "private-signing-secret";
    await user.type(screen.getByLabelText(/Signing secret/), supplied);
    await user.click(screen.getByRole("button", { name: "Create" }));
    expect(
      await screen.findByText(/supplied secret was not displayed/),
    ).toBeInTheDocument();
    expect(screen.queryByText(supplied)).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Done" }));
    await user.click(screen.getByRole("button", { name: "Add event source" }));
    await user.click(screen.getByText("advanced settings"));
    await user.type(screen.getByLabelText(/Signing secret/), supplied);
    await user.click(screen.getByRole("button", { name: "Create" }));
    expect(await screen.findByText(/already registered/)).toBeInTheDocument();
    expect(screen.queryByText(supplied)).not.toBeInTheDocument();
    expect(
      JSON.stringify(
        client
          .getMutationCache()
          .getAll()
          .map((mutation) => mutation.state),
      ),
    ).not.toContain(supplied);
  });

  it("loads beyond the first 50 records", async () => {
    const user = userEvent.setup();
    const offsets: number[] = [];
    server.use(
      http.get("*/api/automation/v1/webhooks", ({ request }) => {
        const offset = Number(new URL(request.url).searchParams.get("offset"));
        offsets.push(offset);
        return HttpResponse.json({
          webhooks: Array.from({ length: offset === 0 ? 50 : 1 }, (_, i) => ({
            id: `${offset + i}`,
            name: `Source ${offset + i}`,
            source: `source-${offset + i}`,
            webhook_url: `https://example.test/${offset + i}`,
            enabled: true,
          })),
          total: 51,
        });
      }),
    );
    renderEvents();
    await user.click(
      await screen.findByRole("button", { name: "Load more event sources" }),
    );
    expect(await screen.findByText("Source 50")).toBeInTheDocument();
    expect(offsets).toEqual([0, 50]);
  });

  it("shows a distinct unsupported state for an unavailable webhook endpoint", async () => {
    server.use(
      http.get(
        "*/api/automation/v1/webhooks",
        () => new HttpResponse(null, { status: 404 }),
      ),
    );
    renderEvents();
    expect(
      await screen.findByText(/Custom webhooks are not supported/),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Add event source" }),
    ).not.toBeInTheDocument();
  });

  it("retries transient list failures and distinguishes unsupported event delivery", async () => {
    const user = userEvent.setup();
    let fails = true;
    server.use(
      http.get("*/api/automation/v1/webhooks", () =>
        fails
          ? new HttpResponse(null, { status: 503 })
          : HttpResponse.json({ webhooks: [], total: 0 }),
      ),
      http.get("*/api/automation/v1/capabilities", () =>
        HttpResponse.json({
          ready: true,
          triggerKinds: ["cron"],
          eventSources: [],
          eventTypes: [],
          triggers: {},
          features: [],
        }),
      ),
    );
    renderEvents();
    expect(
      await screen.findByText("Could not load event sources."),
    ).toBeInTheDocument();
    fails = false;
    await user.click(screen.getByRole("button", { name: /Retry/i }));
    expect(
      await screen.findByText("No custom webhook event sources yet."),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/does not enable event-triggered automations/),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Add event source" }),
    ).toBeEnabled();
  });

  it("gates cloud deep links until permissions resolve and rejects members without calling webhooks", async () => {
    setActiveSelection({ backendId: cloud.id, orgId: "org-member" });
    let resolvePermission!: () => void;
    const permission = new Promise<void>((resolve) => {
      resolvePermission = resolve;
    });
    const requested: string[] = [];
    server.use(
      http.get("*/api/organizations/org-member/me", async () => {
        await permission;
        return HttpResponse.json({
          org_id: "org-member",
          user_id: "user",
          role: "member",
          permissions: ["view_automations"],
        });
      }),
      http.get("*/api/automation/v1/webhooks", ({ request }) => {
        requested.push(request.url);
        return HttpResponse.json({ webhooks: [], total: 0 });
      }),
    );
    renderEvents();
    expect(screen.getByText("Loading event sources…")).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Add event source" }),
    ).not.toBeInTheDocument();
    resolvePermission();
    expect(
      await screen.findByText(
        /Only organization admins and owners can view and manage event sources/,
      ),
    ).toBeInTheDocument();
    await waitFor(() => expect(requested).toEqual([]));
    expect(
      screen.queryByTestId("automations-navigation-events"),
    ).not.toBeInTheDocument();
  });

  it("honors a server-side 403 even when the cloud organization role is admin", async () => {
    setActiveSelection({ backendId: cloud.id, orgId: "org-admin" });
    let requests = 0;
    server.use(
      http.get("*/api/organizations/org-admin/me", () =>
        HttpResponse.json({
          org_id: "org-admin",
          user_id: "user",
          role: "admin",
          permissions: ["manage_automations"],
        }),
      ),
      http.get("*/api/automation/v1/webhooks", () => {
        requests += 1;
        return new HttpResponse(null, { status: 403 });
      }),
    );
    renderEvents();
    expect(
      await screen.findByText(
        /Only organization admins and owners can view and manage event sources/,
      ),
    ).toBeInTheDocument();
    expect(requests).toBe(1);
  });

  it("keeps webhook lists isolated by cloud organization on selection change", async () => {
    setActiveSelection({ backendId: cloud.id, orgId: "org-a" });
    const seen: string[] = [];
    server.use(
      http.get("*/api/organizations/:org/me", ({ params }) =>
        HttpResponse.json({
          org_id: params.org,
          user_id: "user",
          role: "owner",
          permissions: ["manage_automations"],
        }),
      ),
      http.get("*/api/automation/v1/webhooks", ({ request }) => {
        const org = request.headers.get("X-Org-Id") ?? "none";
        seen.push(org);
        return HttpResponse.json({
          webhooks: [
            {
              id: org,
              name: `Source ${org}`,
              source: org,
              webhook_url: `https://example.test/${org}`,
              enabled: true,
            },
          ],
          total: 1,
        });
      }),
    );
    const client = renderEvents();
    expect(await screen.findByText("Source org-a")).toBeInTheDocument();
    act(() => setActiveSelection({ backendId: cloud.id, orgId: "org-b" }));
    expect(await screen.findByText("Source org-b")).toBeInTheDocument();
    expect(screen.queryByText("Source org-a")).not.toBeInTheDocument();
    expect(seen).toEqual(["org-a", "org-b"]);
    expect(
      client.getQueryCache().findAll({ queryKey: ["automation-webhooks"] }),
    ).toHaveLength(2);
  });
});
