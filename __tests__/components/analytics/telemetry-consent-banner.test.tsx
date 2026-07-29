import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { TelemetryConsentBanner } from "#/components/features/analytics/telemetry-consent-banner";

const useActiveBackendMock = vi.fn();
const useSettingsMock = vi.fn();
const useBackendsHealthMock = vi.fn();
const saveSettingsMock = vi.fn();
const getLockedCloudHostMock = vi.fn();
const setTelemetryConsentMock = vi.fn();

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    ready: true,
    t: (key: string, options?: Record<string, string>) =>
      key === "TELEMETRY$BACKEND_SCOPE"
        ? `This preference is saved for ${options?.name} at ${options?.host}.`
        : key,
  }),
}));

vi.mock("#/i18n", () => ({
  OPENHANDS_I18N_NAMESPACE: "openhands",
}));

vi.mock("#/api/agent-server-config", () => ({
  getLockedCloudHost: () => getLockedCloudHostMock(),
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => useActiveBackendMock(),
}));

vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => useSettingsMock(),
}));

vi.mock("#/hooks/query/use-backends-health", () => ({
  useBackendsHealth: (...args: unknown[]) => useBackendsHealthMock(...args),
}));

vi.mock("#/hooks/mutation/use-save-settings", () => ({
  useSaveSettings: () => ({
    mutateAsync: saveSettingsMock,
    isPending: false,
  }),
}));

vi.mock("#/services/telemetry", () => ({
  setTelemetryConsent: (...args: unknown[]) => setTelemetryConsentMock(...args),
}));

describe("TelemetryConsentBanner", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    saveSettingsMock.mockResolvedValue(undefined);
    getLockedCloudHostMock.mockReturnValue(null);
    useActiveBackendMock.mockReturnValue({
      backend: {
        id: "local",
        kind: "local",
        name: "Local Dev",
        host: "http://localhost:12000",
      },
    });
    useBackendsHealthMock.mockReturnValue({
      local: { isConnected: true },
    });

    useSettingsMock.mockReturnValue({
      data: { user_consents_to_analytics: null },
      isSuccess: true,
    });
  });

  it("renders in local mode when agent-server consent is null", async () => {
    render(<TelemetryConsentBanner />);

    expect(
      await screen.findByTestId("telemetry-consent-form"),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        "This preference is saved for Local Dev at http://localhost:12000.",
      ),
    ).toBeInTheDocument();
  });

  it("does not render until the local backend settings are connected", async () => {
    useSettingsMock.mockReturnValue({
      data: { user_consents_to_analytics: null },
      isSuccess: false,
    });

    render(<TelemetryConsentBanner />);

    await waitFor(() => {
      expect(
        screen.queryByTestId("telemetry-consent-form"),
      ).not.toBeInTheDocument();
    });
  });

  it("does not render when the active local backend is unhealthy", async () => {
    useBackendsHealthMock.mockReturnValue({
      local: { isConnected: false },
    });
    useSettingsMock.mockReturnValue({
      data: { user_consents_to_analytics: null },
      isSuccess: true,
    });

    render(<TelemetryConsentBanner />);

    await waitFor(() => {
      expect(
        screen.queryByTestId("telemetry-consent-form"),
      ).not.toBeInTheDocument();
    });
  });

  it.each([true, false])(
    "does not render when agent-server consent is %s",
    async (serverConsent) => {
      useSettingsMock.mockReturnValue({
        data: { user_consents_to_analytics: serverConsent },
        isSuccess: true,
      });

      render(<TelemetryConsentBanner />);

      await waitFor(() => {
        expect(
          screen.queryByTestId("telemetry-consent-form"),
        ).not.toBeInTheDocument();
      });
    },
  );

  it("does not render for cloud backends", async () => {
    useActiveBackendMock.mockReturnValue({
      backend: { id: "cloud", kind: "cloud" },
    });

    render(<TelemetryConsentBanner />);

    await waitFor(() => {
      expect(
        screen.queryByTestId("telemetry-consent-form"),
      ).not.toBeInTheDocument();
    });
  });

  it("does not render in locked Cloud mode", async () => {
    getLockedCloudHostMock.mockReturnValue("https://app.all-hands.dev");

    render(<TelemetryConsentBanner />);

    await waitFor(() => {
      expect(
        screen.queryByTestId("telemetry-consent-form"),
      ).not.toBeInTheDocument();
    });
  });

  it("persists the user's local-mode consent choice without useTelemetry", async () => {
    const user = userEvent.setup();
    render(<TelemetryConsentBanner />);

    const checkbox = await screen.findByRole("checkbox");
    await user.click(checkbox);
    await user.click(screen.getByTestId("confirm-telemetry-preferences"));

    expect(setTelemetryConsentMock).toHaveBeenCalledWith("denied");
    expect(saveSettingsMock).toHaveBeenCalledWith({
      user_consents_to_analytics: false,
    });
  });

  it("does not persist browser consent when the backend save fails", async () => {
    const user = userEvent.setup();
    saveSettingsMock.mockRejectedValue(new Error("unauthorized"));
    render(<TelemetryConsentBanner />);

    await screen.findByTestId("telemetry-consent-form");
    await user.click(screen.getByTestId("confirm-telemetry-preferences"));

    await waitFor(() => {
      expect(saveSettingsMock).toHaveBeenCalledWith({
        user_consents_to_analytics: true,
      });
    });
    expect(setTelemetryConsentMock).not.toHaveBeenCalled();
    expect(screen.getByTestId("telemetry-consent-form")).toBeInTheDocument();
  });

});
