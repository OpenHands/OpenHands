import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { TelemetryConsentBanner } from "#/components/features/analytics/telemetry-consent-banner";
import type { Backend } from "#/api/backend-registry/types";

const localBackend: Backend = {
  id: "local",
  name: "Local",
  host: "http://localhost:3000",
  apiKey: "",
  kind: "local",
} as Backend;

const mocks = vi.hoisted(() => ({
  ready: true,
  backend: null as Backend | null,
  lockedCloudHost: null as string | null,
  health: {} as Record<string, { isConnected: boolean }>,
  settings: null as { user_consents_to_analytics: boolean | null } | null,
  hasLoadedSettings: true,
  isSavingSettings: false,
  saveSettings: vi.fn<(args: unknown) => Promise<void>>(),
  setTelemetryConsent: vi.fn<(value: string) => Promise<void>>(),
  translate: vi.fn<(key: string, params?: Record<string, string>) => string>(),
  useTranslation: vi.fn<(namespace: string) => void>(),
}));

const translations: Record<string, string> = {
  TELEMETRY$CONSENT_TITLE: "Help improve OpenHands",
  TELEMETRY$CONSENT_DESCRIPTION:
    "We collect anonymous usage data to improve the product.",
  TELEMETRY$SEND_ANONYMOUS_DATA: "Send anonymous usage data",
  TELEMETRY$CONFIRM_PREFERENCES: "Confirm preferences",
  TELEMETRY$BACKEND_SCOPE: "Scope:",
};

vi.mock("react-i18next", () => ({
  useTranslation: (namespace: string) => {
    mocks.useTranslation(namespace);
    return { t: mocks.translate, ready: mocks.ready };
  },
}));

vi.mock("#/api/agent-server-config", () => ({
  getLockedCloudHost: () => mocks.lockedCloudHost,
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({ backend: mocks.backend }),
}));

vi.mock("#/hooks/query/use-backends-health", () => ({
  useBackendsHealth: () => mocks.health,
}));

vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => ({
    data: mocks.settings,
    isSuccess: mocks.hasLoadedSettings,
  }),
}));

vi.mock("#/hooks/mutation/use-save-settings", () => ({
  useSaveSettings: () => ({
    mutateAsync: mocks.saveSettings,
    isPending: mocks.isSavingSettings,
  }),
}));

vi.mock("#/services/telemetry", () => ({
  setTelemetryConsent: (value: string) => mocks.setTelemetryConsent(value),
}));

interface RenderOptions {
  onChoice?: (granted: boolean) => void;
}

function prime() {
  vi.useFakeTimers();
  vi.clearAllMocks();
  mocks.ready = true;
  mocks.backend = localBackend;
  mocks.lockedCloudHost = null;
  mocks.health = { local: { isConnected: true } };
  mocks.settings = { user_consents_to_analytics: null };
  mocks.hasLoadedSettings = true;
  mocks.isSavingSettings = false;
  mocks.saveSettings.mockResolvedValue(undefined);
  mocks.setTelemetryConsent.mockResolvedValue(undefined);
  mocks.translate.mockImplementation(
    (key, params) =>
      (translations[key] ?? `Missing:${key}`) +
      (params ? ` ${Object.values(params).join(" ")}` : ""),
  );
}

function renderBanner({ onChoice }: RenderOptions = {}) {
  prime();
  return render(<TelemetryConsentBanner onChoice={onChoice} />);
}

function advanceBy(ms: number) {
  act(() => {
    vi.advanceTimersByTime(ms);
  });
}

async function submitForm() {
  const form = screen.getByTestId("telemetry-consent-form");
  const event = new Event("submit", { bubbles: true, cancelable: true });
  await act(async () => {
    form.dispatchEvent(event);
  });
  return event;
}

afterEach(() => {
  vi.clearAllTimers();
  vi.useRealTimers();
  vi.clearAllMocks();
});

describe("telemetry consent banner", () => {
  it("renders nothing when a cloud host is locked", () => {
    prime();
    mocks.lockedCloudHost = "https://locked.example.com";
    render(<TelemetryConsentBanner />);
    advanceBy(50);
    expect(
      screen.queryByTestId("telemetry-consent-form"),
    ).not.toBeInTheDocument();
  });

  it("renders nothing for a non-local backend", () => {
    prime();
    mocks.backend = { ...localBackend, kind: "cloud" } as Backend;
    render(<TelemetryConsentBanner />);
    advanceBy(50);
    expect(
      screen.queryByTestId("telemetry-consent-form"),
    ).not.toBeInTheDocument();
  });

  it.each([
    [
      "the backend is not connected",
      () => {
        mocks.health = { local: { isConnected: false } };
      },
    ],
    [
      "the backend has no health entry",
      () => {
        mocks.health = {};
      },
    ],
    [
      "settings loaded but data is null",
      () => {
        mocks.hasLoadedSettings = true;
        mocks.settings = null;
      },
    ],
    [
      "settings have not loaded",
      () => {
        mocks.hasLoadedSettings = false;
      },
    ],
    [
      "a consent choice was already made",
      () => {
        mocks.settings = { user_consents_to_analytics: true };
      },
    ],
    [
      "translations are not ready",
      () => {
        mocks.ready = false;
      },
    ],
  ])("stays hidden when %s", (_reason, mutate) => {
    prime();
    mutate();
    render(<TelemetryConsentBanner />);
    advanceBy(50);
    expect(
      screen.queryByTestId("telemetry-consent-form"),
    ).not.toBeInTheDocument();
  });

  it("waits 50 ms before revealing the translated preferences", () => {
    renderBanner();

    expect(
      screen.queryByTestId("telemetry-consent-form"),
    ).not.toBeInTheDocument();
    advanceBy(49);
    expect(
      screen.queryByTestId("telemetry-consent-form"),
    ).not.toBeInTheDocument();

    advanceBy(1);

    expect(screen.getByTestId("telemetry-consent-form")).toBeInTheDocument();
    expect(screen.getByText("Help improve OpenHands")).toBeInTheDocument();
    expect(
      screen.getByText(
        "We collect anonymous usage data to improve the product.",
      ),
    ).toBeInTheDocument();
    const checkbox = screen.getByRole("checkbox", {
      name: "Send anonymous usage data",
    });
    expect(checkbox).toHaveAttribute("name", "analytics");
    expect(checkbox).toBeChecked();
    // The scope line names the specific backend the consent applies to.
    expect(
      screen.getByText("Scope: Local http://localhost:3000"),
    ).toBeInTheDocument();
    expect(mocks.useTranslation).toHaveBeenCalledWith("openhands");
  });

  it("re-prompts after the active backend changes", async () => {
    const onChoice = vi.fn();
    prime();
    const { rerender } = render(<TelemetryConsentBanner onChoice={onChoice} />);
    advanceBy(50);
    await submitForm();
    expect(
      screen.queryByTestId("telemetry-consent-form"),
    ).not.toBeInTheDocument();

    // Switching to a different backend clears the submitted flag so the new
    // backend gets its own consent prompt.
    mocks.backend = { ...localBackend, id: "local-2" } as Backend;
    mocks.health = { "local-2": { isConnected: true } };
    rerender(<TelemetryConsentBanner onChoice={onChoice} />);
    advanceBy(50);
    expect(screen.getByTestId("telemetry-consent-form")).toBeInTheDocument();
  });

  it("grants consent, persists it, and reports the choice on submit", async () => {
    const onChoice = vi.fn();
    renderBanner({ onChoice });
    advanceBy(50);

    const event = await submitForm();

    expect(event.defaultPrevented).toBe(true);
    expect(mocks.saveSettings).toHaveBeenCalledExactlyOnceWith({
      user_consents_to_analytics: true,
    });
    expect(mocks.setTelemetryConsent).toHaveBeenCalledExactlyOnceWith(
      "granted",
    );
    expect(onChoice).toHaveBeenCalledExactlyOnceWith(true);
    // The prompt hides after a choice is submitted.
    expect(
      screen.queryByTestId("telemetry-consent-form"),
    ).not.toBeInTheDocument();
  });

  it("denies consent when the checkbox is unchecked", async () => {
    const onChoice = vi.fn();
    renderBanner({ onChoice });
    advanceBy(50);

    fireEvent.click(
      screen.getByRole("checkbox", { name: "Send anonymous usage data" }),
    );
    const event = await submitForm();

    expect(event.defaultPrevented).toBe(true);
    expect(mocks.saveSettings).toHaveBeenCalledExactlyOnceWith({
      user_consents_to_analytics: false,
    });
    expect(mocks.setTelemetryConsent).toHaveBeenCalledExactlyOnceWith("denied");
    expect(onChoice).toHaveBeenCalledExactlyOnceWith(false);
  });

  it("does not persist telemetry consent or report a choice when saving fails", async () => {
    const onChoice = vi.fn();
    mocks.saveSettings.mockRejectedValueOnce(new Error("save failed"));
    renderBanner({ onChoice });
    advanceBy(50);

    await submitForm();

    expect(mocks.setTelemetryConsent).not.toHaveBeenCalled();
    expect(onChoice).not.toHaveBeenCalled();
    // The prompt remains visible so the user can retry.
    expect(screen.getByTestId("telemetry-consent-form")).toBeInTheDocument();
  });

  it("submits safely without an onChoice callback", async () => {
    renderBanner();
    advanceBy(50);

    const event = await submitForm();

    expect(event.defaultPrevented).toBe(true);
    expect(mocks.saveSettings).toHaveBeenCalledOnce();
    expect(mocks.setTelemetryConsent).toHaveBeenCalledExactlyOnceWith(
      "granted",
    );
  });

  it("disables the confirm button while settings are saving", () => {
    prime();
    mocks.isSavingSettings = true;
    render(<TelemetryConsentBanner />);
    advanceBy(50);

    expect(
      screen.getByTestId("confirm-telemetry-preferences"),
    ).toBeDisabled();
  });
});
