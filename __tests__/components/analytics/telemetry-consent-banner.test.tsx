import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { TelemetryConsentBanner } from "#/components/features/analytics/telemetry-consent-banner";

const useActiveBackendMock = vi.fn();
const useSettingsMock = vi.fn();
const saveSettingsMock = vi.fn();
const getLockedCloudHostMock = vi.fn();
const setTelemetryConsentMock = vi.fn();

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    ready: true,
    t: (key: string) => key,
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

vi.mock("#/hooks/mutation/use-save-settings", () => ({
  useSaveSettings: () => ({ mutate: saveSettingsMock }),
}));

vi.mock("#/services/telemetry", () => ({
  setTelemetryConsent: (...args: unknown[]) => setTelemetryConsentMock(...args),
}));

describe("TelemetryConsentBanner", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    getLockedCloudHostMock.mockReturnValue(null);
    useActiveBackendMock.mockReturnValue({
      backend: { id: "local", kind: "local" },
    });
    useSettingsMock.mockReturnValue({
      data: { user_consents_to_analytics: null },
    });
  });

  it("renders in local mode when agent-server consent is null", async () => {
    render(<TelemetryConsentBanner />);

    expect(
      await screen.findByTestId("telemetry-consent-form"),
    ).toBeInTheDocument();
  });

  it.each([true, false])(
    "does not render when agent-server consent is %s",
    async (serverConsent) => {
      useSettingsMock.mockReturnValue({
        data: { user_consents_to_analytics: serverConsent },
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
});
