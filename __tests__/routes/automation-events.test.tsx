import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router";
import { http, HttpResponse } from "msw";
import { server } from "#/mocks/node";
import { I18nKey } from "#/i18n/declaration";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import type { Backend } from "#/api/backend-registry/types";
import AutomationEvents, { clientLoader } from "#/routes/automation-events";

// The pinned extensions release predates Events; exercise the real host admission
// against a candidate with the same four Events fields as the next release.
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
        events: {
          title: "Event sources",
          description: "Manage custom webhook event sources.",
        },
      },
      endpoints: { ...base.endpoints, webhooks: "/v1/webhooks" },
      navigation: {
        ...base.navigation,
        subPages: [
          ...(base.navigation.subPages ?? []),
          { page: "events", label: "Events", icon: "activity" },
        ],
      },
    },
  };
});

const localBackend: Backend = {
  id: "local-1",
  name: "Local",
  host: "http://localhost:8000",
  apiKey: "session-key",
  kind: "local",
};
const cloudBackend: Backend = {
  id: "cloud-1",
  name: "Cloud",
  host: "https://app.all-hands.dev",
  apiKey: "test-bearer-key",
  kind: "cloud",
};
const webhook = {
  id: "webhook-1",
  org_id: "org-1",
  name: "Issues",
  source: "issues",
  webhook_url: "https://example.com/v1/events/org-1/issues",
  event_key_expr: "type",
  signature_header: "X-Signature-256",
  signature_scheme: "hmac_sha256",
  enabled: true,
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
};

function renderEvents() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <ActiveBackendProvider>
        <MemoryRouter initialEntries={["/automations/events"]}>
          <AutomationEvents />
        </MemoryRouter>
      </ActiveBackendProvider>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  setRegisteredBackends([localBackend]);
  setActiveSelection({ backendId: localBackend.id });
});

afterEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
});

describe("automation Events page", () => {
  it("admits the declared route, shows the reachability notice, and lists paginated sources", async () => {
    server.use(
      http.get("*/api/automation/v1/webhooks", ({ request }) => {
        const offset = Number(new URL(request.url).searchParams.get("offset"));
        return HttpResponse.json({
          webhooks:
            offset === 0
              ? [webhook]
              : [{ ...webhook, id: "webhook-2", name: "Comments" }],
          total: 2,
        });
      }),
    );
    expect(clientLoader()).toBeNull();
    renderEvents();
    expect(screen.getByRole("note")).toHaveTextContent(
      I18nKey.AUTOMATIONS$EVENTS$REACHABILITY_WARNING,
    );
    expect(await screen.findByText("Issues")).toBeInTheDocument();
    expect(screen.getByText(webhook.webhook_url)).toBeInTheDocument();
    const header = screen.getByTestId("automation-events-header");
    expect(
      within(header).getByRole("heading", { name: "Event sources" }),
    ).toBeInTheDocument();
    const createButton = within(header).getByRole("button", {
      name: I18nKey.AUTOMATIONS$EVENTS$ADD,
    });
    expect(createButton).toHaveClass("h-9", "bg-base-secondary");
    expect(createButton).toHaveAttribute("aria-expanded", "false");
    await userEvent.click(createButton);
    expect(
      within(header).getByRole("button", { name: I18nKey.BUTTON$CANCEL }),
    ).toHaveAttribute("aria-expanded", "true");
    await userEvent.click(
      within(header).getByRole("button", { name: I18nKey.BUTTON$CANCEL }),
    );
    await userEvent.click(
      screen.getByRole("button", {
        name: I18nKey.AUTOMATIONS$EVENTS$LOAD_MORE,
      }),
    );
    expect(await screen.findByText("Comments")).toBeInTheDocument();
  });

  it("retries a failed list request and displays the resulting sources", async () => {
    let requests = 0;
    server.use(
      http.get("*/api/automation/v1/webhooks", () => {
        requests += 1;
        return requests < 3
          ? HttpResponse.json(
              { detail: "Service unavailable" },
              { status: 503 },
            )
          : HttpResponse.json({ webhooks: [webhook], total: 1 });
      }),
    );
    renderEvents();
    expect(
      await screen.findByRole("alert", {}, { timeout: 3000 }),
    ).toHaveTextContent(I18nKey.AUTOMATIONS$EVENTS$FETCH_ERROR);
    await userEvent.click(
      screen.getByRole("button", { name: I18nKey.AUTOMATIONS$ERROR_RETRY }),
    );
    expect(await screen.findByText("Issues")).toBeInTheDocument();
  });

  it("creates a source and shows a generated secret only until dismissed", async () => {
    const requests: unknown[] = [];
    let created = false;
    server.use(
      http.get("*/api/automation/v1/webhooks", () =>
        HttpResponse.json({
          webhooks: created ? [webhook] : [],
          total: created ? 1 : 0,
        }),
      ),
      http.post("*/api/automation/v1/webhooks", async ({ request }) => {
        requests.push(await request.json());
        created = true;
        return HttpResponse.json(
          { ...webhook, webhook_secret: "generated-signing-secret" },
          { status: 201 },
        );
      }),
    );
    renderEvents();
    expect(
      await screen.findByText(I18nKey.AUTOMATIONS$EVENTS$EMPTY),
    ).toBeInTheDocument();
    await userEvent.click(
      screen.getByRole("button", { name: I18nKey.AUTOMATIONS$EVENTS$ADD }),
    );
    await userEvent.type(
      screen.getByRole("textbox", {
        name: I18nKey.AUTOMATION_SETUP$CUSTOM_WEBHOOK_NAME,
      }),
      "Issues",
    );
    await userEvent.type(
      screen.getByRole("textbox", { name: /AUTOMATIONS\$EVENTS\$SOURCE/ }),
      "issues",
    );
    await userEvent.click(
      screen.getByRole("button", {
        name: I18nKey.AUTOMATIONS$EVENTS$ADD,
      }),
    );
    expect(
      await screen.findByText("generated-signing-secret"),
    ).toBeInTheDocument();
    expect(requests).toEqual([
      {
        name: "Issues",
        source: "issues",
        event_key_expr: "type",
        signature_header: "X-Signature-256",
      },
    ]);
    await userEvent.click(
      screen.getByRole("button", { name: I18nKey.AUTOMATIONS$EVENTS$DONE }),
    );
    expect(
      screen.queryByText("generated-signing-secret"),
    ).not.toBeInTheDocument();
    expect(await screen.findByText("Issues")).toBeInTheDocument();
  });

  it("shows server validation errors without creating a source", async () => {
    server.use(
      http.get("*/api/automation/v1/webhooks", () =>
        HttpResponse.json({ webhooks: [], total: 0 }),
      ),
      http.post("*/api/automation/v1/webhooks", () =>
        HttpResponse.json(
          {
            detail: [
              {
                loc: ["body", "source"],
                msg: "Source is reserved",
                type: "value_error",
              },
            ],
          },
          { status: 422 },
        ),
      ),
    );
    renderEvents();
    await screen.findByText(I18nKey.AUTOMATIONS$EVENTS$EMPTY);
    await userEvent.click(
      screen.getByRole("button", { name: I18nKey.AUTOMATIONS$EVENTS$ADD }),
    );
    await userEvent.type(
      screen.getByRole("textbox", {
        name: I18nKey.AUTOMATION_SETUP$CUSTOM_WEBHOOK_NAME,
      }),
      "Issues",
    );
    await userEvent.type(
      screen.getByRole("textbox", { name: /AUTOMATIONS\$EVENTS\$SOURCE/ }),
      "github",
    );
    await userEvent.click(
      screen.getByRole("button", { name: I18nKey.AUTOMATIONS$EVENTS$ADD }),
    );
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Source is reserved",
    );
    expect(screen.queryByText("Issues")).not.toBeInTheDocument();
  });

  it("submits a supplied secret without displaying it after creation", async () => {
    let submitted: unknown;
    server.use(
      http.get("*/api/automation/v1/webhooks", () =>
        HttpResponse.json({ webhooks: [], total: 0 }),
      ),
      http.post("*/api/automation/v1/webhooks", async ({ request }) => {
        submitted = await request.json();
        return HttpResponse.json(webhook, { status: 201 });
      }),
    );
    renderEvents();
    await screen.findByText(I18nKey.AUTOMATIONS$EVENTS$EMPTY);
    await userEvent.click(
      screen.getByRole("button", { name: I18nKey.AUTOMATIONS$EVENTS$ADD }),
    );
    await userEvent.type(
      screen.getByRole("textbox", {
        name: I18nKey.AUTOMATION_SETUP$CUSTOM_WEBHOOK_NAME,
      }),
      "Issues",
    );
    await userEvent.type(
      screen.getByRole("textbox", { name: /AUTOMATIONS\$EVENTS\$SOURCE/ }),
      "issues",
    );
    await userEvent.type(
      screen.getByLabelText(/AUTOMATION_SETUP\$CUSTOM_WEBHOOK_SECRET/),
      "customer-secret-value",
    );
    await userEvent.click(
      screen.getByRole("button", {
        name: I18nKey.AUTOMATIONS$EVENTS$ADD,
      }),
    );
    await screen.findByText(I18nKey.AUTOMATIONS$EVENTS$CREATED_WITH_OWN_SECRET);
    expect(submitted).toMatchObject({
      webhook_secret: "customer-secret-value",
    });
    expect(screen.queryByText("customer-secret-value")).not.toBeInTheDocument();
  });

  it("clears the one-time secret when switching backend scope", async () => {
    server.use(
      http.get("*/api/automation/v1/webhooks", () =>
        HttpResponse.json({ webhooks: [], total: 0 }),
      ),
      http.post("*/api/automation/v1/webhooks", () =>
        HttpResponse.json(
          { ...webhook, webhook_secret: "generated-signing-secret" },
          { status: 201 },
        ),
      ),
      http.get("*/api/organizations/org-1/me", () =>
        HttpResponse.json({
          org_id: "org-1",
          user_id: "user-1",
          role: "member",
          permissions: ["view_automations"],
        }),
      ),
    );
    setRegisteredBackends([localBackend, cloudBackend]);
    renderEvents();
    await screen.findByText(I18nKey.AUTOMATIONS$EVENTS$EMPTY);
    await userEvent.click(
      screen.getByRole("button", { name: I18nKey.AUTOMATIONS$EVENTS$ADD }),
    );
    await userEvent.type(
      screen.getByRole("textbox", {
        name: I18nKey.AUTOMATION_SETUP$CUSTOM_WEBHOOK_NAME,
      }),
      "Issues",
    );
    await userEvent.type(
      screen.getByRole("textbox", { name: /AUTOMATIONS\$EVENTS\$SOURCE/ }),
      "issues",
    );
    await userEvent.click(
      screen.getByRole("button", {
        name: I18nKey.AUTOMATIONS$EVENTS$ADD,
      }),
    );
    await screen.findByText("generated-signing-secret");

    act(() =>
      setActiveSelection({ backendId: cloudBackend.id, orgId: "org-1" }),
    );
    expect(
      screen.queryByText("generated-signing-secret"),
    ).not.toBeInTheDocument();
    expect(
      await screen.findByText(I18nKey.AUTOMATIONS$EVENTS$EMPTY),
    ).toBeInTheDocument();
  });

  it("shows webhook results to readers without offering creation", async () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id, orgId: "org-1" });
    server.use(
      http.get("*/api/organizations/org-1/me", () =>
        HttpResponse.json({
          org_id: "org-1",
          user_id: "user-1",
          role: "member",
          permissions: ["view_automations"],
        }),
      ),
      http.get("*/api/automation/v1/webhooks", () =>
        HttpResponse.json({ webhooks: [webhook], total: 1 }),
      ),
    );
    renderEvents();
    expect(await screen.findByText("Issues")).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: I18nKey.AUTOMATIONS$EVENTS$ADD }),
    ).not.toBeInTheDocument();
  });

  it.each([403, 404])(
    "does not expose create controls when the service returns %i",
    async (status) => {
      server.use(
        http.get(
          "*/api/automation/v1/webhooks",
          () => new HttpResponse(null, { status }),
        ),
      );
      renderEvents();
      const alert = await screen.findByRole("alert");
      expect(alert).toHaveTextContent(
        status === 404
          ? I18nKey.AUTOMATIONS$EVENTS$UNSUPPORTED
          : I18nKey.AUTOMATIONS$EVENTS$FETCH_ERROR,
      );
      expect(
        screen.queryByRole("button", { name: I18nKey.AUTOMATIONS$EVENTS$ADD }),
      ).not.toBeInTheDocument();
    },
  );
});
