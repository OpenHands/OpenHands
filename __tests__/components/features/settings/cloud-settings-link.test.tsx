import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { CloudSettingsLink } from "#/components/features/settings/cloud-settings-link";

const cloudBackend: Backend = {
  id: "cloud-1",
  name: "OpenHands Cloud",
  host: "https://app.all-hands.dev",
  apiKey: "token",
  kind: "cloud",
};

const localBackend: Backend = {
  id: "local-1",
  name: "Local",
  host: "http://localhost:3001",
  apiKey: "local-key",
  kind: "local",
};

function renderWithProviders() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <ActiveBackendProvider>
        <CloudSettingsLink />
      </ActiveBackendProvider>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
});

afterEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
});

describe("CloudSettingsLink", () => {
  it("renders a link to {cloudHost}/settings when a cloud backend is active", () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });

    renderWithProviders();

    const link = screen.getByTestId("settings-cloud-link");
    expect(link).toHaveAttribute("href", "https://app.all-hands.dev/settings");
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
    // The test i18n mock returns the key verbatim; the human label
    // ("Cloud") is exercised via the translation.json entry instead.
    expect(link).toHaveTextContent("SETTINGS$CLOUD_SETTINGS_LINK");
  });

  it("strips a trailing slash when building the cloud settings URL", () => {
    setRegisteredBackends([
      { ...cloudBackend, host: "https://app.all-hands.dev/" },
    ]);
    setActiveSelection({ backendId: cloudBackend.id });

    renderWithProviders();

    expect(screen.getByTestId("settings-cloud-link")).toHaveAttribute(
      "href",
      "https://app.all-hands.dev/settings",
    );
  });

  it("renders nothing when a local backend is active", () => {
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });

    renderWithProviders();

    expect(screen.queryByTestId("settings-cloud-link")).toBeNull();
  });

  it("renders nothing when no backend is configured", () => {
    renderWithProviders();

    expect(screen.queryByTestId("settings-cloud-link")).toBeNull();
  });
});
