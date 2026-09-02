import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { IntegrationsSettingsLink } from "#/components/features/settings/integrations-settings-link";

const cloudBackend: Backend = {
  id: "cloud-1",
  name: "OpenHands Cloud",
  host: "https://app.all-hands.dev",
  apiKey: "token",
  kind: "cloud",
};

function renderWithProviders() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <ActiveBackendProvider>
        <IntegrationsSettingsLink />
      </ActiveBackendProvider>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
});

afterEach(() => {
  vi.unstubAllEnvs();
  window.localStorage.clear();
  __resetActiveStoreForTests();
});

describe("IntegrationsSettingsLink", () => {
  it("renders a link to {cloudHost}/settings/integrations when a cloud backend is active", () => {
    // Arrange
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });

    // Act
    renderWithProviders();

    // Assert
    const link = screen.getByTestId("settings-integrations-link");
    expect(link).toHaveAttribute(
      "href",
      "https://app.all-hands.dev/settings/integrations",
    );
    expect(link).toHaveAttribute("target", "_blank");
  });

  it("appends the active org so cloud integrations open on the same organization", () => {
    // Arrange
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id, orgId: "org-123" });

    // Act
    renderWithProviders();

    // Assert
    expect(screen.getByTestId("settings-integrations-link")).toHaveAttribute(
      "href",
      "https://app.all-hands.dev/settings/integrations?org=org-123",
    );
  });

  it("renders nothing when the canvas is locked to a Cloud host", () => {
    // Arrange — SaaS / self-hosted OHE serve the canvas with `--lock-to-cloud`;
    // the OHE settings shell already exposes Integrations.
    vi.stubEnv("VITE_LOCK_TO_CLOUD", "https://app.all-hands.dev");
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });

    // Act
    renderWithProviders();

    // Assert
    expect(screen.queryByTestId("settings-integrations-link")).toBeNull();
  });
});
