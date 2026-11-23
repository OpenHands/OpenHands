import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import type { LicenseStatus } from "#/api/cloud/license-service.api";
import { callCloudProxy } from "#/api/cloud/proxy";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { LicenseBanner } from "#/components/features/alerts/license-banner";

vi.mock("#/api/cloud/proxy", () => ({ callCloudProxy: vi.fn() }));

const CLOUD_HOST = "https://ohe.example.com";

const cloudBackend: Backend = {
  id: "cloud-1",
  name: "OpenHands Enterprise",
  host: CLOUD_HOST,
  apiKey: "token",
  kind: "cloud",
};

function licenseStatus(overrides: Partial<LicenseStatus>): LicenseStatus {
  return {
    state: "valid",
    reason: null,
    offline: false,
    type: "paid",
    license_ends_at: null,
    expires_at: "2026-11-21T00:00:00Z",
    last_checked_at: "2026-11-07T00:00:00Z",
    customer_center_url: "https://customercenter.openhands.dev",
    ...overrides,
  };
}

let queryClient: QueryClient;

function renderBanner() {
  queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <ActiveBackendProvider>
        <LicenseBanner />
      </ActiveBackendProvider>
    </QueryClientProvider>,
  );
}

/** An OHE-hosted Canvas: locked to the cloud host that is also the active backend. */
function lockToActiveCloud() {
  vi.stubEnv("VITE_LOCK_TO_CLOUD", CLOUD_HOST);
  setRegisteredBackends([cloudBackend]);
  setActiveSelection({ backendId: cloudBackend.id });
}

async function waitForLicenseRequest() {
  await waitFor(() => {
    expect(callCloudProxy).toHaveBeenCalledTimes(1);
    expect(queryClient.isFetching()).toBe(0);
  });
}

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  vi.mocked(callCloudProxy).mockReset();
});

afterEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  vi.unstubAllEnvs();
});

describe("LicenseBanner", () => {
  it("shows the trial end date and a Customer Center link when the instance is on a trial", async () => {
    // Arrange
    lockToActiveCloud();
    vi.mocked(callCloudProxy).mockResolvedValue(
      licenseStatus({ type: "trial", license_ends_at: "2026-11-21T00:00:00Z" }),
    );

    // Act
    renderBanner();

    // Assert
    const banner = await screen.findByTestId("license-banner");
    expect(callCloudProxy).toHaveBeenCalledWith(
      expect.objectContaining({ method: "GET", path: "/api/license/status" }),
    );
    expect(banner).toHaveAttribute("role", "status");
    expect(banner).toHaveTextContent("LICENSE$TRIAL_ENDS");
    expect(
      screen.getByRole("link", { name: "LICENSE$OPEN_CUSTOMER_CENTER" }),
    ).toHaveAttribute("href", "https://customercenter.openhands.dev");
  });

  it("alerts when the instance has no license key", async () => {
    // Arrange
    lockToActiveCloud();
    vi.mocked(callCloudProxy).mockResolvedValue(
      licenseStatus({ state: "missing_key", type: null }),
    );

    // Act
    renderBanner();

    // Assert
    const banner = await screen.findByTestId("license-banner");
    expect(banner).toHaveAttribute("role", "alert");
    expect(banner).toHaveTextContent("LICENSE$MISSING_KEY");
  });

  it("renders nothing when the instance has license enforcement off", async () => {
    // Arrange
    lockToActiveCloud();
    vi.mocked(callCloudProxy).mockResolvedValue(
      licenseStatus({
        state: "disabled",
        type: null,
        expires_at: null,
        last_checked_at: null,
      }),
    );

    // Act
    renderBanner();
    await waitForLicenseRequest();

    // Assert
    expect(screen.queryByTestId("license-banner")).toBeNull();
  });

  it("renders nothing when the instance does not serve the license route", async () => {
    // Arrange: an older instance answers with its web app instead of JSON
    lockToActiveCloud();
    vi.mocked(callCloudProxy).mockResolvedValue(
      "<!doctype html><title>OpenHands</title>",
    );

    // Act
    renderBanner();
    await waitForLicenseRequest();

    // Assert
    expect(screen.queryByTestId("license-banner")).toBeNull();
  });

  it("does not ask for the license when Canvas is not locked to the instance", async () => {
    // Arrange: a standalone Canvas with a cloud backend added by hand
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });

    // Act
    renderBanner();
    await waitFor(() => expect(queryClient.isFetching()).toBe(0));

    // Assert
    expect(callCloudProxy).not.toHaveBeenCalled();
    expect(screen.queryByTestId("license-banner")).toBeNull();
  });
});
