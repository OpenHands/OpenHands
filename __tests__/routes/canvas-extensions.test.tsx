import i18next from "i18next";
import translations from "#/i18n/translation.json";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";
import CanvasExtensionsScreen from "#/routes/canvas-extensions";
import CanvasExtensionsService from "#/api/canvas-extensions-service";
import { SecretsService } from "#/api/secrets-service";
import SettingsService from "#/api/settings-service/settings-service.api";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";
import { server } from "#/mocks/node";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import type { InstalledCanvasExtensionInfo } from "#/types/canvas-extension";

vi.mock("react-i18next", async (importOriginal) => ({
  ...(await importOriginal<typeof import("react-i18next")>()),
  useTranslation: () => ({ t: i18next.getFixedT("en", "app-secret-test") }),
}));

vi.mock("#/context/navigation-context", () => ({
  useNavigation: () => ({
    navigate: vi.fn(),
    currentPath: "/apps",
    conversationId: null,
    isNavigating: false,
  }),
  NavigationProvider: ({ children }: { children: React.ReactNode }) => children,
}));

const localBackend: Backend = {
  id: "local",
  name: "Local",
  host: "http://127.0.0.1:8001",
  apiKey: "",
  kind: "local",
};

const cloudBackend: Backend = {
  id: "cloud",
  name: "Enterprise",
  host: "https://app.example.test",
  apiKey: "",
  kind: "cloud",
};

/**
 * Selects an organization on a cloud backend whose instance reports
 * `appsEnabled` and grants the current user `permissions`.
 */
function selectCloudOrganization({
  appsEnabled,
  permissions,
}: {
  appsEnabled: boolean;
  permissions: string[];
}) {
  setRegisteredBackends([cloudBackend]);
  setActiveSelection({ backendId: cloudBackend.id, orgId: "org-1" });
  server.use(
    http.get("*/api/v1/web-client/config", () =>
      HttpResponse.json({
        providers_configured: [],
        feature_flags: { enable_canvas_apps: appsEnabled },
      }),
    ),
    http.get("*/api/organizations/org-1/me", () =>
      HttpResponse.json({
        org_id: "org-1",
        user_id: "user-1",
        role: "member",
        permissions,
      }),
    ),
  );
}

function buildExtension(
  overrides: Partial<InstalledCanvasExtensionInfo> = {},
): InstalledCanvasExtensionInfo {
  return {
    name: "demo-page",
    version: "0.1.0",
    enabled: true,
    source: "https://github.com/OpenHands/OpenHands",
    repo_path: "src/fixtures/canvas-extensions/demo-page",
    installed_at: "2026-10-01T00:00:00Z",
    install_path: "/home/.openhands/canvas-extensions/installed/demo-page",
    ...overrides,
  };
}

function renderAppsScreen() {
  return render(<CanvasExtensionsScreen />, {
    wrapper: ({ children }) => (
      <QueryClientProvider
        client={
          new QueryClient({ defaultOptions: { queries: { retry: false } } })
        }
      >
        <ActiveBackendProvider>{children}</ActiveBackendProvider>
      </QueryClientProvider>
    ),
  });
}

/** Starts an Update whose install request stays pending until resolved. */
async function startPendingUpdate(user: ReturnType<typeof userEvent.setup>) {
  let finishUpdate: () => void = () => {};
  vi.spyOn(CanvasExtensionsService, "install").mockImplementation(
    () =>
      new Promise((resolve) => {
        finishUpdate = () => resolve(buildExtension());
      }),
  );
  await user.click(
    await screen.findByTestId("canvas-extension-refresh-demo-page"),
  );
  await waitFor(() =>
    expect(
      screen.getByTestId("canvas-extension-uninstall-demo-page"),
    ).toBeDisabled(),
  );
  return () => finishUpdate();
}

describe("CanvasExtensionsScreen", () => {
  beforeEach(async () => {
    await i18next.init({
      lng: "en",
      resources: {
        en: {
          "app-secret-test": Object.fromEntries(
            Object.entries(translations).map(([key, value]) => [key, value.en]),
          ),
        },
      },
    });
    vi.restoreAllMocks();
    __resetActiveStoreForTests();
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      MOCK_DEFAULT_USER_SETTINGS,
    );
  });

  it("links missing managed-App secrets to the mounted Canvas settings", async () => {
    const user = userEvent.setup();
    vi.stubEnv("VITE_BASE_PATH", "/canvas");
    vi.spyOn(SecretsService, "getSecrets").mockResolvedValue([]);
    vi.spyOn(CanvasExtensionsService, "backendStatus").mockResolvedValue({
      name: "demo-page",
      state: "stopped",
      revision: "revision",
    });
    vi.spyOn(CanvasExtensionsService, "listInstalled").mockResolvedValue([
      buildExtension({
        enabled: false,
        manifest: {
          schema_version: 1,
          name: "demo-page",
          display_name: "Demo page",
          version: "0.1.0",
          entrypoint: "dist/extension.js",
          backend: { secret_refs: ["MISSING_KEY"] },
        },
      }),
    ]);
    try {
      renderAppsScreen();
      await user.click(await screen.findByRole("switch"));
      const alert = await screen.findByRole("alert");
      expect(alert).toHaveTextContent("Required server secrets: MISSING_KEY");
      expect(alert.textContent).not.toContain("{{names}}");
      expect(
        screen.getByRole("link", { name: "Secrets settings" }),
      ).toHaveAttribute("href", "/canvas/settings/secrets");
    } finally {
      vi.unstubAllEnvs();
    }
  });

  // Issue #17956: the switch was only styled inert while busy, so the
  // keyboard could still disable an app in the middle of an Update.
  it("keeps an enabled app's switch inert to the keyboard while an Update is pending", async () => {
    const user = userEvent.setup();
    vi.spyOn(CanvasExtensionsService, "listInstalled").mockResolvedValue([
      buildExtension({ enabled: true }),
    ]);
    const setEnabledSpy = vi
      .spyOn(CanvasExtensionsService, "setEnabled")
      .mockResolvedValue(buildExtension({ enabled: false }));
    renderAppsScreen();

    const finishUpdate = await startPendingUpdate(user);
    const appSwitch = screen.getByRole("switch");
    expect(appSwitch).toBeDisabled();
    appSwitch.focus();
    expect(appSwitch).not.toHaveFocus();
    await user.keyboard(" ");
    expect(setEnabledSpy).not.toHaveBeenCalled();

    finishUpdate();
    await waitFor(() => expect(appSwitch).toBeEnabled());
    appSwitch.focus();
    await user.keyboard(" ");
    await waitFor(() =>
      expect(setEnabledSpy).toHaveBeenCalledWith(
        "demo-page",
        false,
        undefined,
        localBackend,
      ),
    );
  });

  it("does not open the enable confirmation from a disabled app's switch while an Update is pending", async () => {
    const user = userEvent.setup();
    vi.spyOn(CanvasExtensionsService, "listInstalled").mockResolvedValue([
      buildExtension({ enabled: false }),
    ]);
    renderAppsScreen();

    const finishUpdate = await startPendingUpdate(user);
    const appSwitch = screen.getByRole("switch");
    appSwitch.focus();
    await user.keyboard(" ");
    expect(screen.queryByTestId("confirmation-modal")).not.toBeInTheDocument();

    finishUpdate();
    await waitFor(() => expect(appSwitch).toBeEnabled());
    appSwitch.focus();
    await user.keyboard(" ");
    expect(screen.getByTestId("confirmation-modal")).toBeInTheDocument();
  });

  describe("on a cloud backend", () => {
    it("lets a member switch approved Apps without the controls that manage them", async () => {
      // Arrange
      selectCloudOrganization({
        appsEnabled: true,
        permissions: ["view_org_settings"],
      });
      vi.spyOn(CanvasExtensionsService, "listInstalled").mockResolvedValue([
        buildExtension({ enabled: false }),
      ]);

      // Act
      renderAppsScreen();

      // Assert
      expect(await screen.findByRole("switch")).toBeEnabled();
      expect(
        screen.queryByTestId("canvas-extensions-add-button"),
      ).not.toBeInTheDocument();
      expect(
        screen.queryByTestId("canvas-extension-refresh-demo-page"),
      ).not.toBeInTheDocument();
      expect(
        screen.queryByTestId("canvas-extension-uninstall-demo-page"),
      ).not.toBeInTheDocument();
    });

    it("lets a Super Admin add Apps and remove one for the organization", async () => {
      // Arrange
      const user = userEvent.setup();
      selectCloudOrganization({
        appsEnabled: true,
        permissions: ["manage_super_admins"],
      });
      vi.spyOn(CanvasExtensionsService, "listInstalled").mockResolvedValue([
        buildExtension(),
      ]);
      const uninstallSpy = vi
        .spyOn(CanvasExtensionsService, "uninstall")
        .mockResolvedValue({ message: "uninstalled" });
      renderAppsScreen();

      // Act
      await user.click(
        await screen.findByTestId("canvas-extension-uninstall-demo-page"),
      );
      await user.click(screen.getByTestId("confirm-button"));

      // Assert
      await waitFor(() =>
        expect(uninstallSpy).toHaveBeenCalledWith("demo-page"),
      );
      expect(screen.getByTestId("canvas-extensions-add-button")).toBeEnabled();
      // An App is replaced by removing it and adding the new revision.
      expect(
        screen.queryByTestId("canvas-extension-refresh-demo-page"),
      ).not.toBeInTheDocument();
    });

    it("keeps Apps unavailable while the instance has them turned off", async () => {
      // Arrange
      selectCloudOrganization({
        appsEnabled: false,
        permissions: ["manage_super_admins"],
      });
      const listSpy = vi
        .spyOn(CanvasExtensionsService, "listInstalled")
        .mockResolvedValue([buildExtension()]);

      // Act
      renderAppsScreen();

      // Assert
      expect(
        await screen.findByText(
          "Apps are not available on Cloud backends yet.",
        ),
      ).toBeInTheDocument();
      expect(listSpy).not.toHaveBeenCalled();
    });

    it("drops an App removed for the organization while the page stays open", async () => {
      // Arrange
      vi.useFakeTimers({ toFake: ["setInterval", "clearInterval"] });
      try {
        selectCloudOrganization({ appsEnabled: true, permissions: [] });
        let approved = [buildExtension()];
        vi.spyOn(CanvasExtensionsService, "listInstalled").mockImplementation(
          async () => approved,
        );
        renderAppsScreen();
        await screen.findByTestId("canvas-extension-card-demo-page");

        // Act
        approved = [];
        act(() => {
          vi.advanceTimersByTime(60_000);
        });

        // Assert
        await waitFor(() =>
          expect(
            screen.queryByTestId("canvas-extension-card-demo-page"),
          ).not.toBeInTheDocument(),
        );
      } finally {
        vi.useRealTimers();
      }
    });
  });
});
