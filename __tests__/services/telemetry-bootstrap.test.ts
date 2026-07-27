import { beforeEach, describe, expect, it, vi } from "vitest";

type MockPostHog = {
  init: ReturnType<typeof vi.fn>;
  capture: ReturnType<typeof vi.fn>;
  captureException: ReturnType<typeof vi.fn>;
  opt_in_capturing: ReturnType<typeof vi.fn>;
  opt_out_capturing: ReturnType<typeof vi.fn>;
  has_opted_out_capturing: ReturnType<typeof vi.fn>;
  identify: ReturnType<typeof vi.fn>;
  get_property: ReturnType<typeof vi.fn>;
  reset: ReturnType<typeof vi.fn>;
};

async function loadTelemetryWithLegacyUser() {
  let identifiedUserId: string | undefined = "legacy-cloud-user";
  const mockPosthog: MockPostHog = {
    init: vi.fn(),
    capture: vi.fn(),
    captureException: vi.fn(),
    opt_in_capturing: vi.fn(),
    opt_out_capturing: vi.fn(),
    has_opted_out_capturing: vi.fn(() => false),
    identify: vi.fn((userId: string) => {
      identifiedUserId = userId;
    }),
    get_property: vi.fn((property: string) =>
      property === "$user_id" ? identifiedUserId : undefined,
    ),
    reset: vi.fn(() => {
      identifiedUserId = undefined;
    }),
  };
  mockPosthog.init.mockReturnValue(mockPosthog);

  vi.doMock("posthog-js", () => ({
    default: mockPosthog,
  }));

  const telemetry = await import("#/services/telemetry");
  return { telemetry, mockPosthog };
}

describe("Telemetry bootstrap identity migration", () => {
  beforeEach(() => {
    vi.resetModules();
    vi.clearAllMocks();
    localStorage.clear();
    sessionStorage.clear();
    delete (window as unknown as Record<string, unknown>)
      .__AGENT_CANVAS_LOCK_TO_CLOUD__;
  });

  it("clears a legacy identified user when no OAuth bootstrap is present", async () => {
    const { telemetry, mockPosthog } = await loadTelemetryWithLegacyUser();

    await telemetry.setTelemetryConsent("granted");

    expect(mockPosthog.reset).toHaveBeenCalledWith(false);
    expect(mockPosthog.opt_in_capturing).toHaveBeenCalled();
    expect(mockPosthog.identify).not.toHaveBeenCalled();
  });

  it("preserves OAuth bootstrap identity instead of resetting legacy users", async () => {
    const { telemetry, mockPosthog } = await loadTelemetryWithLegacyUser();

    telemetry.configurePostHogBootstrap({
      distinctID: "bootstrapped-browser-id",
      sessionID: "bootstrapped-session-id",
    });
    await telemetry.setTelemetryConsent("granted");

    expect(mockPosthog.reset).not.toHaveBeenCalled();
    expect(mockPosthog.init).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        bootstrap: {
          distinctID: "bootstrapped-browser-id",
          sessionID: "bootstrapped-session-id",
        },
        person_profiles: "always",
      }),
      "agent-canvas",
    );
    expect(mockPosthog.identify).not.toHaveBeenCalled();
  });
});
