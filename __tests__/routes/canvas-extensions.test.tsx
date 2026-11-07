import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import CanvasExtensionsScreen from "#/routes/canvas-extensions";
import CanvasExtensionsService from "#/api/canvas-extensions-service";
import SettingsService from "#/api/settings-service/settings-service.api";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import type { InstalledCanvasExtensionInfo } from "#/types/canvas-extension";

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
  beforeEach(() => {
    vi.restoreAllMocks();
    __resetActiveStoreForTests();
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      MOCK_DEFAULT_USER_SETTINGS,
    );
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
      expect(setEnabledSpy).toHaveBeenCalledWith("demo-page", false),
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
});
