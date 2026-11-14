import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TelemetryConsentBanner } from "#/components/features/analytics/telemetry-consent-banner";
import SettingsService from "#/api/settings-service/settings-service.api";
import { server } from "#/mocks/node";

// The banner, `useSettings` and `SettingsService` run for real; only the
// agent-server answer to GET /api/settings, health and i18n are stubbed.
vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key, ready: true }),
}));

vi.mock("#/hooks/query/use-backends-health", () => ({
  useBackendsHealth: (backends: { id: string }[]) =>
    Object.fromEntries(
      backends.map((backend) => [backend.id, { isConnected: true }]),
    ),
}));

let settingsResponse: "unreachable" | boolean | null = "unreachable";

const settingsHandler = http.get("*/api/settings", ({ request }) => {
  if (new URL(request.url).pathname !== "/api/settings") return undefined;
  if (settingsResponse === "unreachable") return HttpResponse.error();
  return HttpResponse.json({
    agent_settings: {},
    conversation_settings: {},
    llm_api_key_is_set: false,
    misc_settings: {
      app_preferences: { user_consents_to_analytics: settingsResponse },
    },
  });
});

async function advanceBy(ms: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

function renderBanner() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  render(
    <QueryClientProvider client={queryClient}>
      <TelemetryConsentBanner />
    </QueryClientProvider>,
  );
}

describe("telemetry consent banner after a failed settings fetch", () => {
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    vi.spyOn(console, "warn").mockImplementation(() => {});
    SettingsService.invalidateCache();
    settingsResponse = "unreachable";
    server.use(settingsHandler);
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
    SettingsService.invalidateCache();
  });

  it("never asks for consent the backend already holds while it is unreachable or after it answers", async () => {
    // Arrange: the first settings read happens while the agent server is down.
    renderBanner();
    let sawConsentForm = false;
    const watchFor = async (ms: number) => {
      for (let elapsed = 0; elapsed < ms; elapsed += 2_000) {
        await advanceBy(2_000);
        sawConsentForm ||= !!screen.queryByTestId("telemetry-consent-form");
      }
    };
    await watchFor(4_000);

    // Act: the agent server comes back holding the user's earlier answer.
    settingsResponse = false;
    await watchFor(30_000);

    // Assert
    expect(sawConsentForm).toBe(false);
  });

  it("asks once the recovered backend reports consent as unanswered", async () => {
    // Arrange
    renderBanner();
    await advanceBy(3_000);
    expect(
      screen.queryByTestId("telemetry-consent-form"),
    ).not.toBeInTheDocument();

    // Act
    settingsResponse = null;
    await advanceBy(30_000);

    // Assert
    expect(screen.getByTestId("telemetry-consent-form")).toBeInTheDocument();
  });
});
