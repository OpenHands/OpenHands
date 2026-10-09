import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import type { CloudSetupGuideSteps } from "#/api/cloud/types";
import SuperAdminSetupGuide from "#/components/features/setup-guide/super-admin-setup-guide";
import type { SuperAdminSetupStepId } from "#/components/features/setup-guide/super-admin-setup-guide.constants";
import { notifySuperAdminSetupStep } from "#/components/features/setup-guide/super-admin-setup-step-event";
import { startSetupGuideTour } from "#/components/features/setup-guide/tour/setup-guide-tour";
import { NavigationProvider } from "#/context/navigation-context";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { I18nKey } from "#/i18n/declaration";
import { server } from "#/mocks/node";

const locked = vi.hoisted(() => ({ cloudHost: null as string | null }));
const tour = vi.hoisted(() => ({ active: false }));

// Canvas served from the cloud host opens enterprise pages in its own tab.
vi.mock("#/api/agent-server-config", async (importOriginal) => ({
  ...(await importOriginal<typeof import("#/api/agent-server-config")>()),
  getLockedCloudHost: () => locked.cloudHost,
}));

// The spotlight needs a real layout; these tests check which tour starts.
vi.mock(
  "#/components/features/setup-guide/tour/setup-guide-tour",
  async (importOriginal) => ({
    ...(await importOriginal<
      typeof import("#/components/features/setup-guide/tour/setup-guide-tour")
    >()),
    startSetupGuideTour: vi.fn(),
  }),
);

vi.mock("#/components/features/setup-guide/tour/tour-engine", () => ({
  startGuidedTour: vi.fn(),
  isGuidedTourActive: () => tour.active,
  subscribeGuidedTourActive: () => () => {},
}));

const CLOUD_HOST = "https://ohe.example.com";
const GUIDE_ORG_ID = "org-1";
/** The invite step on the guide's organization, asking for its tour. */
const INVITE_TOUR_URL = `${CLOUD_HOST}/settings/org-members?org=${GUIDE_ORG_ID}&setup_tour=invite-users`;

const cloudBackend: Backend = {
  id: "cloud-1",
  name: "OHE",
  host: CLOUD_HOST,
  apiKey: "",
  kind: "cloud",
};

const localBackend: Backend = {
  id: "local-1",
  name: "Local",
  host: "http://localhost:8000",
  apiKey: "local-key",
  kind: "local",
};

function guideState(steps: Partial<CloudSetupGuideSteps> = {}) {
  return {
    wizard_pending: false,
    guide_org_id: GUIDE_ORG_ID,
    guide_dismissed: false,
    guide_steps: {
      org_llm: false,
      mcp_server: false,
      automation: false,
      invite: false,
      ...steps,
    },
  };
}

/**
 * Serves `/me` and the setup state for the cloud host and counts the
 * requests, so cases can assert who asks the server for a guide.
 */
function serveSetupGuide({
  permissions = ["manage_super_admins"],
  setupState = () => HttpResponse.json(guideState()),
}: {
  permissions?: string[];
  setupState?: () => Response;
} = {}) {
  const requests = { me: 0, setupState: 0 };
  server.use(
    http.get("*/api/organizations/:orgId/me", ({ params }) => {
      requests.me += 1;
      return HttpResponse.json({
        org_id: params.orgId,
        user_id: "user-1",
        role: "owner",
        permissions,
      });
    }),
    http.get("*/api/admin/setup-state", () => {
      requests.setupState += 1;
      return setupState();
    }),
  );
  return requests;
}

function activateBackend(backend: Backend) {
  setRegisteredBackends([backend]);
  setActiveSelection({ backendId: backend.id, orgId: GUIDE_ORG_ID });
}

function renderGuide(currentPath = "/conversations") {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const navigate = vi.fn();
  const ui = (path: string) => (
    <QueryClientProvider client={queryClient}>
      <ActiveBackendProvider>
        <NavigationProvider
          value={{
            currentPath: path,
            conversationId: null,
            isNavigating: false,
            navigate,
          }}
        >
          <SuperAdminSetupGuide />
        </NavigationProvider>
      </ActiveBackendProvider>
    </QueryClientProvider>
  );
  const view = render(ui(currentPath));
  return { navigate, navigateTo: (path: string) => view.rerender(ui(path)) };
}

async function expectNoGuide() {
  await expect(
    screen.findByTestId("super-admin-setup-guide", undefined, {
      timeout: 200,
    }),
  ).rejects.toThrow();
}

describe("SuperAdminSetupGuide", () => {
  beforeEach(() => {
    localStorage.clear();
    __resetActiveStoreForTests();
    locked.cloudHost = null;
    tour.active = false;
    vi.mocked(startSetupGuideTour).mockReset();
  });

  afterEach(() => {
    __resetActiveStoreForTests();
    vi.unstubAllGlobals();
    window.history.replaceState(null, "", "/");
  });

  it("shows the first Super Admin the progress the server reports", async () => {
    activateBackend(cloudBackend);
    serveSetupGuide({
      setupState: () => HttpResponse.json(guideState({ org_llm: true })),
    });

    renderGuide();

    expect(
      await screen.findByTestId("super-admin-setup-guide-panel"),
    ).toBeInTheDocument();
    expect(screen.getByText("1/4")).toBeInTheDocument();
    expect(screen.getByRole("progressbar")).toHaveAttribute(
      "aria-valuenow",
      "25",
    );
  });

  it("links each step to the page where it is done", async () => {
    activateBackend(cloudBackend);
    serveSetupGuide();

    renderGuide();

    await screen.findByTestId("super-admin-setup-guide-panel");
    expect(
      screen.getByTestId("super-admin-setup-guide-step-add-llm"),
    ).toHaveAttribute(
      "href",
      `${CLOUD_HOST}/settings/org-defaults?org=${GUIDE_ORG_ID}`,
    );
    expect(
      screen.getByTestId("super-admin-setup-guide-step-add-integration"),
    ).toHaveAttribute("href", "/mcp");
    expect(
      screen.getByTestId("super-admin-setup-guide-step-invite-users"),
    ).toHaveAttribute(
      "href",
      `${CLOUD_HOST}/settings/org-members?org=${GUIDE_ORG_ID}`,
    );
    expect(
      screen.getByTestId("super-admin-setup-guide-step-optional-saml"),
    ).toHaveAttribute("href", `${CLOUD_HOST}/super-admin/instance`);
    expect(
      screen.getByTestId("super-admin-setup-guide-step-first-automation"),
    ).toHaveAttribute("href", "/automations/templates");
    expect(screen.getByTestId("super-admin-setup-guide-page")).toHaveAttribute(
      "href",
      `${CLOUD_HOST}/super-admin/setup`,
    );
  });

  it("opens the next step from Start", async () => {
    activateBackend(cloudBackend);
    serveSetupGuide({
      setupState: () => HttpResponse.json(guideState({ org_llm: true })),
    });

    renderGuide();

    expect(
      await screen.findByTestId("super-admin-setup-guide-start"),
    ).toHaveAttribute("href", "/automations/templates");
  });

  it("opens the Canvas MCP page from Start for the integration step", async () => {
    activateBackend(cloudBackend);
    serveSetupGuide({
      setupState: () =>
        HttpResponse.json(guideState({ org_llm: true, automation: true })),
    });

    renderGuide();

    expect(
      await screen.findByTestId("super-admin-setup-guide-start"),
    ).toHaveAttribute("href", "/mcp");
  });

  it("opens an enterprise step from Start on the guide's organization, with its tour", async () => {
    activateBackend(cloudBackend);
    serveSetupGuide({
      setupState: () =>
        HttpResponse.json(
          guideState({ org_llm: true, automation: true, mcp_server: true }),
        ),
    });

    renderGuide();

    expect(
      await screen.findByTestId("super-admin-setup-guide-start"),
    ).toHaveAttribute("href", INVITE_TOUR_URL);
  });

  describe("when the guide's next step is done", () => {
    it("opens the Canvas MCP page with its tour after an automation is created", async () => {
      activateBackend(cloudBackend);
      let state = guideState({ org_llm: true });
      serveSetupGuide({ setupState: () => HttpResponse.json(state) });
      const { navigate } = renderGuide("/automations/new/widget");
      expect(await screen.findByText("1/4")).toBeInTheDocument();

      state = guideState({ org_llm: true, automation: true });
      act(() => notifySuperAdminSetupStep("first-automation"));

      // The tour opens the page itself.
      await waitFor(() =>
        expect(startSetupGuideTour).toHaveBeenCalledWith(
          "add-integration",
          expect.objectContaining({ navigate }),
          expect.any(Function),
        ),
      );
      expect(navigate).not.toHaveBeenCalled();
    });

    it("opens the enterprise invite page with its tour when Canvas is served from the cloud host", async () => {
      locked.cloudHost = CLOUD_HOST;
      const assign = vi.fn();
      vi.stubGlobal("location", { ...window.location, assign });
      activateBackend(cloudBackend);
      let state = guideState({ org_llm: true, automation: true });
      serveSetupGuide({ setupState: () => HttpResponse.json(state) });
      renderGuide("/mcp");
      expect(await screen.findByText("2/4")).toBeInTheDocument();

      state = guideState({ org_llm: true, automation: true, mcp_server: true });
      act(() => notifySuperAdminSetupStep("add-integration"));

      await waitFor(() => expect(assign).toHaveBeenCalledWith(INVITE_TOUR_URL));
    });

    it("points at Start instead of opening the enterprise page in a new tab", async () => {
      const assign = vi.fn();
      vi.stubGlobal("location", { ...window.location, assign });
      const user = userEvent.setup();
      activateBackend(cloudBackend);
      let state = guideState({ org_llm: true, automation: true });
      serveSetupGuide({ setupState: () => HttpResponse.json(state) });
      renderGuide("/mcp");
      expect(await screen.findByText("2/4")).toBeInTheDocument();
      await user.click(
        screen.getByRole("button", { name: I18nKey.BUTTON$CLOSE }),
      );

      state = guideState({ org_llm: true, automation: true, mcp_server: true });
      act(() => notifySuperAdminSetupStep("add-integration"));

      expect(
        await screen.findByTestId("super-admin-setup-guide-start"),
      ).toHaveAttribute("href", INVITE_TOUR_URL);
      expect(assign).not.toHaveBeenCalled();
    });
  });

  it.each<[string, Partial<CloudSetupGuideSteps>, SuperAdminSetupStepId]>([
    ["a step finished out of order", { mcp_server: true }, "add-integration"],
    ["a step the server does not report done", {}, "first-automation"],
  ])("opens nothing after %s", async (_case, after, completed) => {
    const assign = vi.fn();
    vi.stubGlobal("location", { ...window.location, assign });
    activateBackend(cloudBackend);
    let state = guideState({ org_llm: true });
    const requests = serveSetupGuide({
      setupState: () => HttpResponse.json(state),
    });
    const { navigate } = renderGuide("/automations/templates");
    expect(await screen.findByText("1/4")).toBeInTheDocument();
    const before = requests.setupState;

    state = guideState({ org_llm: true, ...after });
    act(() => notifySuperAdminSetupStep(completed));

    await waitFor(() => expect(requests.setupState).toBeGreaterThan(before));
    await act(async () => {});
    expect(navigate).not.toHaveBeenCalled();
    expect(assign).not.toHaveBeenCalled();
  });

  it("closes the panel from Start, even on the next step's own page", async () => {
    activateBackend(cloudBackend);
    serveSetupGuide({
      setupState: () => HttpResponse.json(guideState({ org_llm: true })),
    });
    const user = userEvent.setup();
    renderGuide("/automations/templates");

    await user.click(
      await screen.findByTestId("super-admin-setup-guide-start"),
    );

    expect(
      screen.queryByTestId("super-admin-setup-guide-panel"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("super-admin-setup-guide-toggle"),
    ).toHaveAttribute("aria-expanded", "false");
  });

  it("starts a Canvas step's tour from Start, which opens the page itself", async () => {
    activateBackend(cloudBackend);
    serveSetupGuide({
      setupState: () => HttpResponse.json(guideState({ org_llm: true })),
    });
    const user = userEvent.setup();
    const { navigate } = renderGuide();

    await user.click(
      await screen.findByTestId("super-admin-setup-guide-start"),
    );

    expect(startSetupGuideTour).toHaveBeenCalledWith(
      "first-automation",
      expect.objectContaining({ navigate }),
      expect.any(Function),
    );
    expect(navigate).not.toHaveBeenCalled();
    expect(
      screen.getByTestId("super-admin-setup-guide-toggle"),
    ).toHaveAttribute("aria-expanded", "false");
  });

  it("starts the tour a page opened from the enterprise guide asks for, once", async () => {
    window.history.replaceState(
      null,
      "",
      "/mcp?org=org-1&setup_tour=add-integration",
    );
    activateBackend(cloudBackend);
    serveSetupGuide({
      setupState: () =>
        HttpResponse.json(guideState({ org_llm: true, automation: true })),
    });

    renderGuide("/mcp");

    await waitFor(() =>
      expect(startSetupGuideTour).toHaveBeenCalledWith(
        "add-integration",
        expect.anything(),
        expect.any(Function),
      ),
    );
    expect(startSetupGuideTour).toHaveBeenCalledTimes(1);
    expect(window.location.search).toBe("?org=org-1");
  });

  it("ignores a tour request for a step done in the enterprise app", async () => {
    window.history.replaceState(null, "", "/mcp?setup_tour=invite-users");
    activateBackend(cloudBackend);
    serveSetupGuide();

    renderGuide("/mcp");

    await screen.findByTestId("super-admin-setup-guide-panel");
    await waitFor(() => expect(window.location.search).toBe(""));
    expect(startSetupGuideTour).not.toHaveBeenCalled();
  });

  it("stays out of the way while a tour is open", async () => {
    tour.active = true;
    activateBackend(cloudBackend);
    const requests = serveSetupGuide();

    renderGuide();

    await waitFor(() => expect(requests.setupState).toBeGreaterThan(0));
    await expectNoGuide();
  });

  it("re-reads progress when the admin moves to another page", async () => {
    activateBackend(cloudBackend);
    let state = guideState({ org_llm: true });
    serveSetupGuide({ setupState: () => HttpResponse.json(state) });
    const { navigateTo } = renderGuide("/automations/templates");
    expect(await screen.findByText("1/4")).toBeInTheDocument();

    state = guideState({ org_llm: true, automation: true });
    navigateTo("/automations/automation-1");

    expect(await screen.findByText("2/4")).toBeInTheDocument();
  });

  it("closes to the pill and opens again from it", async () => {
    const user = userEvent.setup();
    activateBackend(cloudBackend);
    serveSetupGuide();
    renderGuide();
    await screen.findByTestId("super-admin-setup-guide-panel");

    await user.click(
      screen.getByRole("button", { name: I18nKey.BUTTON$CLOSE }),
    );
    expect(
      screen.queryByTestId("super-admin-setup-guide-panel"),
    ).not.toBeInTheDocument();

    await user.click(screen.getByTestId("super-admin-setup-guide-toggle"));
    expect(
      screen.getByTestId("super-admin-setup-guide-panel"),
    ).toBeInTheDocument();
  });

  it("does not ask for the setup state for a user who is not a Super Admin", async () => {
    activateBackend(cloudBackend);
    const requests = serveSetupGuide({ permissions: ["view_automations"] });

    renderGuide();

    await waitFor(() => expect(requests.me).toBe(1));
    await expectNoGuide();
    expect(requests.setupState).toBe(0);
  });

  it.each([
    [
      "the caller is not the first Super Admin",
      () =>
        HttpResponse.json({
          wizard_pending: false,
          guide_org_id: null,
          guide_dismissed: false,
          guide_steps: null,
        }),
    ],
    [
      "the guide was dismissed",
      () => HttpResponse.json({ ...guideState(), guide_dismissed: true }),
    ],
    [
      "every step is done",
      () =>
        HttpResponse.json(
          guideState({
            org_llm: true,
            mcp_server: true,
            automation: true,
            invite: true,
          }),
        ),
    ],
    [
      "the server has the Super Admin feature off and serves its web page",
      () => HttpResponse.html("<!doctype html><html></html>"),
    ],
  ])("shows nothing when %s", async (_case, setupState) => {
    activateBackend(cloudBackend);
    const requests = serveSetupGuide({ setupState });

    renderGuide();

    await waitFor(() => expect(requests.setupState).toBeGreaterThan(0));
    await expectNoGuide();
  });

  it("never asks a local backend for a guide", async () => {
    activateBackend(localBackend);
    const requests = serveSetupGuide();

    renderGuide();

    await expectNoGuide();
    expect(requests).toEqual({ me: 0, setupState: 0 });
  });
});
