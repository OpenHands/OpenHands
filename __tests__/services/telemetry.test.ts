import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";

// Mock posthog-js before importing telemetry service
let identifiedUserId: string | undefined;
let anonymousDistinctId = "ph-test-distinct-id";
let latestPostHogConfig:
  | (Record<string, unknown> & { before_send: (event: unknown) => unknown })
  | undefined;
const mockPosthog = {
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
  get_distinct_id: vi.fn(() => identifiedUserId ?? anonymousDistinctId),
  reset: vi.fn((resetDeviceId = false) => {
    identifiedUserId = undefined;
    anonymousDistinctId = resetDeviceId
      ? "ph-reset-device-id"
      : "ph-reset-distinct-id";
  }),
};
mockPosthog.init.mockImplementation((_, config) => {
  latestPostHogConfig = config;
  return mockPosthog;
});

vi.mock("posthog-js", () => ({
  default: mockPosthog,
}));

import {
  clearPendingCloudTelemetryConsent,
  clearPendingLocalTelemetryRevocation,
  clearTelemetryData,
  configureTelemetry,
  getPendingCloudTelemetryConsent,
  getPendingLocalTelemetryRevocationId,
  getTelemetryConsent,
  getTelemetryDistinctId,
  getTelemetryDistinctIdForConsentSync,
  initializePostHogClient,
  isTelemetryEnabled,
  setTelemetryBackendContext,
  setTelemetryCloudContext,
  setTelemetryConsent,
  setTelemetryIdentity,
  subscribeTelemetryConsent,
  trackEvent,
  trackException,
  trackInstall,
  trackSessionStart,
} from "#/services/telemetry";

// Mock import.meta.env for tests
vi.stubGlobal("import.meta", {
  env: {
    DEV: false,
    VITE_DO_NOT_TRACK: undefined,
  },
});

describe("Telemetry Service", () => {
  beforeEach(async () => {
    // Clear localStorage before each test
    localStorage.clear();
    sessionStorage.clear();
    delete (window as unknown as Record<string, unknown>)
      .__AGENT_CANVAS_LOCK_TO_CLOUD__;
    // Reset mock
    vi.clearAllMocks();
    identifiedUserId = undefined;
    anonymousDistinctId = "ph-test-distinct-id";
    mockPosthog.has_opted_out_capturing.mockReturnValue(false);
    mockPosthog.get_distinct_id.mockImplementation(
      () => identifiedUserId ?? anonymousDistinctId,
    );
    await setTelemetryIdentity(null);
    setTelemetryBackendContext({});
  });

  afterEach(() => {
    configureTelemetry({});
    localStorage.clear();
    sessionStorage.clear();
    delete (window as unknown as Record<string, unknown>)
      .__AGENT_CANVAS_LOCK_TO_CLOUD__;
  });

  describe("PostHog ownership", () => {
    it("retries one named Canvas client with runtime configuration", async () => {
      configureTelemetry({
        apiKey: "phc_embedded",
        apiHost: "https://events.example.com",
        uiHost: "https://posthog.example.com",
      });
      configureTelemetry({
        apiKey: undefined,
        apiHost: undefined,
        uiHost: undefined,
      });
      mockPosthog.init.mockImplementationOnce(() => null);
      await setTelemetryConsent("granted");
      await expect(initializePostHogClient()).resolves.toBe(mockPosthog);

      expect(mockPosthog.init).toHaveBeenCalledTimes(2);
      expect(mockPosthog.init).toHaveBeenCalledWith(
        "phc_embedded",
        expect.objectContaining({
          api_host: "https://events.example.com",
          ui_host: "https://posthog.example.com",
          persistence_name: "agent-canvas",
          consent_persistence_name: "agent-canvas-consent",
          person_profiles: "always",
          capture_pageview: "history_change",
          autocapture: true,
        }),
        "agent-canvas",
      );
      expect(mockPosthog.opt_in_capturing).toHaveBeenCalled();
      const config = mockPosthog.init.mock.calls[1][1];
      expect(
        config.before_send({
          event: "test_event",
          properties: { client_source: "incorrect", custom: "value" },
        }),
      ).toEqual({
        event: "test_event",
        properties: expect.objectContaining({
          client_source: "agent_canvas",
          client_version: expect.any(String),
          package_name: "@openhands/agent-canvas",
          package_version: expect.any(String),
          backend_kind: null,
          agent_server_version: "unknown",
          automation_sdk_version: "unknown",
          backend_version: "unknown",
          custom: "value",
        }),
      });

      setTelemetryBackendContext({
        backendKind: "local",
        agentServerVersion: "1.36.2",
        automationSdkVersion: "1.36.3",
      });
      expect(
        config.before_send({
          event: "backend_context_event",
          properties: { backend_kind: "cloud", custom: "value" },
        }),
      ).toEqual({
        event: "backend_context_event",
        properties: expect.objectContaining({
          backend_kind: "cloud",
          agent_server_version: "1.36.2",
          automation_sdk_version: "1.36.3",
          backend_version: "1.36.2",
          custom: "value",
        }),
      });
    });
  });

  describe("Cloud identity and context", () => {
    it("identifies a consented Cloud user and adds Cloud event context", async () => {
      await setTelemetryConsent("granted");

      setTelemetryCloudContext({
        userId: "user-a",
        email: "a@example.com",
        orgId: "org-a",
      });
      await setTelemetryIdentity("user-a", { email: "a@example.com" });

      expect(mockPosthog.identify).toHaveBeenCalledWith("user-a", {
        email: "a@example.com",
      });
      await expect(getTelemetryDistinctId()).resolves.toBe("user-a");
      expect(latestPostHogConfig).toBeDefined();
      expect(
        latestPostHogConfig!.before_send({
          event: "cloud_event",
          properties: {},
        }),
      ).toEqual({
        event: "cloud_event",
        properties: expect.objectContaining({
          cloud_user_id: "user-a",
          cloud_user_email: "a@example.com",
          cloud_org_id: "org-a",
        }),
      });
    });

    it("resets before switching Cloud accounts and restores consent", async () => {
      await setTelemetryConsent("granted");
      await setTelemetryIdentity("user-a");
      vi.clearAllMocks();

      await setTelemetryIdentity("user-b");

      expect(mockPosthog.reset).toHaveBeenCalledWith(false);
      expect(mockPosthog.opt_in_capturing).toHaveBeenCalledOnce();
      expect(mockPosthog.identify).toHaveBeenCalledWith("user-b", {});
    });

    it("clears identity on logout without changing the device", async () => {
      await setTelemetryConsent("granted");
      await setTelemetryIdentity("user-a");
      vi.clearAllMocks();

      await setTelemetryIdentity(null);

      expect(mockPosthog.reset).toHaveBeenCalledWith(false);
      expect(mockPosthog.opt_in_capturing).toHaveBeenCalledOnce();
      expect(mockPosthog.identify).not.toHaveBeenCalled();
    });

    it("removes identity on denial and reapplies it after consent returns", async () => {
      await setTelemetryConsent("granted");
      await setTelemetryIdentity("user-a");
      vi.clearAllMocks();

      await setTelemetryConsent("denied");

      expect(mockPosthog.reset).toHaveBeenCalledWith(false);
      expect(mockPosthog.opt_out_capturing).toHaveBeenCalled();

      vi.clearAllMocks();
      await setTelemetryConsent("granted");

      expect(mockPosthog.identify).toHaveBeenCalledWith("user-a", {});
    });

    it("syncs denial for the actor that existed before PostHog reset", async () => {
      await setTelemetryConsent("granted");
      await setTelemetryIdentity("user-a");

      await setTelemetryConsent("denied");

      expect(mockPosthog.get_distinct_id()).toBe("ph-reset-distinct-id");
      expect(getPendingLocalTelemetryRevocationId()).toBe("user-a");
      await expect(getTelemetryDistinctIdForConsentSync()).resolves.toBe(
        "user-a",
      );

      clearPendingLocalTelemetryRevocation("user-a");
      expect(getPendingLocalTelemetryRevocationId()).toBeNull();
    });

    it("clears Cloud user context for local events", async () => {
      await setTelemetryConsent("granted");

      setTelemetryCloudContext({
        userId: "user-a",
        email: "a@example.com",
        orgId: "org-a",
      });
      setTelemetryCloudContext(null);

      expect(latestPostHogConfig).toBeDefined();
      expect(
        latestPostHogConfig!.before_send({
          event: "local_event",
          properties: {},
        }),
      ).toEqual({
        event: "local_event",
        properties: expect.objectContaining({
          cloud_user_id: null,
          cloud_user_email: null,
          cloud_org_id: null,
        }),
      });
    });
  });

  describe("getTelemetryConsent", () => {
    it("returns 'pending' when no consent has been set", () => {
      expect(getTelemetryConsent()).toBe("pending");
    });

    it("returns 'granted' when consent is granted", () => {
      localStorage.setItem("openhands-telemetry-consent", "granted");
      expect(getTelemetryConsent()).toBe("granted");
    });

    it("returns 'denied' when consent is denied", () => {
      localStorage.setItem("openhands-telemetry-consent", "denied");
      expect(getTelemetryConsent()).toBe("denied");
    });

    it("treats same-origin locked Cloud cookie auth as granted", () => {
      localStorage.setItem("openhands-telemetry-consent", "denied");
      (
        window as unknown as Record<string, unknown>
      ).__AGENT_CANVAS_LOCK_TO_CLOUD__ = window.location.origin;

      expect(getTelemetryConsent()).toBe("granted");
    });
  });

  describe("setTelemetryConsent", () => {
    it("stores granted consent in localStorage", async () => {
      await setTelemetryConsent("granted");
      expect(localStorage.getItem("openhands-telemetry-consent")).toBe(
        "granted",
      );
    });

    it("stores denied consent in localStorage", async () => {
      await setTelemetryConsent("denied");
      expect(localStorage.getItem("openhands-telemetry-consent")).toBe(
        "denied",
      );
    });

    it("applies consent synchronously once the shared client is initialized", async () => {
      await setTelemetryConsent("denied");
      vi.clearAllMocks();

      const update = setTelemetryConsent("granted");

      expect(mockPosthog.opt_in_capturing).toHaveBeenCalledTimes(1);
      await update;
    });

    it("marks an explicit pre-login choice for backend reconciliation", async () => {
      const listener = vi.fn();
      const unsubscribe = subscribeTelemetryConsent(listener);
      await setTelemetryConsent("granted");

      expect(getPendingCloudTelemetryConsent()).toBe("granted");
      expect(listener).toHaveBeenCalledTimes(1);
      unsubscribe();
    });

    it("applies capture consent before notifying identity reconcilers", async () => {
      const listener = vi.fn(() => {
        expect(getTelemetryConsent()).toBe("granted");
        expect(mockPosthog.opt_in_capturing).toHaveBeenCalledTimes(1);
      });
      const unsubscribe = subscribeTelemetryConsent(listener);

      await setTelemetryConsent("granted");

      expect(listener).toHaveBeenCalledTimes(1);
      unsubscribe();
    });

    it("does not mark consent mirrored from backend settings as pending", async () => {
      await setTelemetryConsent("granted", { syncToCloud: false });

      expect(getPendingCloudTelemetryConsent()).toBeNull();
    });

    it("does not re-emit PostHog opt-in for unchanged granted consent", async () => {
      await setTelemetryConsent("granted");
      vi.clearAllMocks();

      await setTelemetryConsent("granted", { syncToCloud: false });

      expect(mockPosthog.opt_in_capturing).not.toHaveBeenCalled();
    });

    it("only clears the pending decision it expects", async () => {
      await setTelemetryConsent("granted");

      clearPendingCloudTelemetryConsent("denied");
      expect(getPendingCloudTelemetryConsent()).toBe("granted");

      clearPendingCloudTelemetryConsent("granted");
      expect(getPendingCloudTelemetryConsent()).toBeNull();
    });
  });

  describe("isTelemetryEnabled", () => {
    it("returns false when consent is pending", () => {
      expect(isTelemetryEnabled()).toBe(false);
    });

    it("returns true when consent is granted", async () => {
      await setTelemetryConsent("granted");
      expect(isTelemetryEnabled()).toBe(true);
    });

    it("returns false when consent is denied", async () => {
      await setTelemetryConsent("denied");
      expect(isTelemetryEnabled()).toBe(false);
    });
  });

  describe("trackInstall", () => {
    it("sends install event immediately without consent (new behavior)", async () => {
      // No consent set - should still send the install event
      await trackInstall();

      expect(mockPosthog.capture).toHaveBeenCalledTimes(1);
      expect(mockPosthog.capture).toHaveBeenCalledWith(
        "canvas_install",
        expect.objectContaining({
          platform: expect.any(String),
          user_agent: expect.any(String),
        }),
      );
    });

    it("only sends install event once", async () => {
      await trackInstall();
      await trackInstall();
      await trackInstall();

      // Should only be called once
      expect(mockPosthog.capture).toHaveBeenCalledTimes(1);
    });

    it("includes correct event data", async () => {
      await trackInstall();

      expect(mockPosthog.capture).toHaveBeenCalledWith(
        "canvas_install",
        expect.objectContaining({
          platform: expect.any(String),
          user_agent: expect.any(String),
          referrer: expect.any(String),
          url_origin: expect.any(String),
          embedded: expect.any(Boolean),
        }),
      );
    });

    it("restores opt-out state after sending install event when consent not granted", async () => {
      // No consent set
      await trackInstall();

      // Should opt out capturing after sending install event
      expect(mockPosthog.opt_out_capturing).toHaveBeenCalled();
    });

    it("does not restore opt-out state when consent is granted", async () => {
      // Grant consent first
      await setTelemetryConsent("granted");
      vi.clearAllMocks(); // Clear the opt_in call from setTelemetryConsent

      await trackInstall();

      // Should NOT call opt_out_capturing when consent is granted
      expect(mockPosthog.opt_out_capturing).not.toHaveBeenCalled();
    });
  });

  describe("getTelemetryDistinctId", () => {
    it("returns null when consent is not granted", async () => {
      await expect(getTelemetryDistinctId()).resolves.toBeNull();
    });

    it("returns the PostHog distinct ID when consent is granted", async () => {
      await setTelemetryConsent("granted");

      await expect(getTelemetryDistinctId()).resolves.toBe(
        "ph-test-distinct-id",
      );
    });
  });

  describe("trackEvent", () => {
    it("does not send event when consent is not granted", async () => {
      await trackEvent("test_event", { foo: "bar" });
      expect(mockPosthog.capture).not.toHaveBeenCalled();
    });

    it("sends custom event when consent is granted", async () => {
      await setTelemetryConsent("granted");
      await trackEvent("custom_action", { button: "submit" });

      expect(mockPosthog.capture).toHaveBeenCalledWith("custom_action", {
        button: "submit",
      });
    });

    it("does not repair SDK opt-out or capture when consent is denied", async () => {
      await setTelemetryConsent("denied");
      vi.clearAllMocks();
      mockPosthog.has_opted_out_capturing.mockReturnValue(true);

      await trackEvent("custom_action");

      expect(mockPosthog.opt_in_capturing).not.toHaveBeenCalled();
      expect(mockPosthog.capture).not.toHaveBeenCalled();
    });

    it("repairs a stale SDK opt-out before a consented custom event", async () => {
      localStorage.setItem("openhands-telemetry-consent", "granted");
      mockPosthog.has_opted_out_capturing.mockReturnValue(true);

      await trackEvent("custom_action");

      expect(mockPosthog.opt_in_capturing).toHaveBeenCalledTimes(1);
      expect(mockPosthog.capture).toHaveBeenCalledWith("custom_action", {});
    });

    it("stops an initialized client when telemetry is disabled", async () => {
      await setTelemetryConsent("granted");
      vi.clearAllMocks();

      configureTelemetry(false);
      await trackEvent("custom_action");

      expect(mockPosthog.opt_out_capturing).toHaveBeenCalledTimes(1);
      expect(mockPosthog.capture).not.toHaveBeenCalled();
      configureTelemetry({});
    });

    it("does not let a consent refresh override a hard disable", async () => {
      await setTelemetryConsent("denied");
      configureTelemetry(false);
      vi.clearAllMocks();

      await setTelemetryConsent("granted", { syncToCloud: false });

      expect(mockPosthog.opt_in_capturing).not.toHaveBeenCalled();
      configureTelemetry({});
    });
  });

  describe("trackSessionStart", () => {
    it("repairs a stale SDK opt-out before recording the session", async () => {
      localStorage.setItem("openhands-telemetry-consent", "granted");
      mockPosthog.has_opted_out_capturing.mockReturnValue(true);

      await trackSessionStart();

      expect(mockPosthog.opt_in_capturing).toHaveBeenCalledTimes(1);
      expect(mockPosthog.capture).toHaveBeenCalledWith(
        "canvas_new_session",
        expect.any(Object),
      );
    });
  });

  describe("trackException", () => {
    it("uses the consent-aware boundary", async () => {
      await setTelemetryConsent("granted");
      const error = new Error("failure");

      await trackException(error, { error_source: "test" });

      expect(mockPosthog.captureException).toHaveBeenCalledWith(error, {
        error_source: "test",
      });
    });
  });

  describe("clearTelemetryData", () => {
    it("clears all telemetry data from localStorage", async () => {
      await setTelemetryConsent("granted");
      await setTelemetryIdentity("user-a");
      localStorage.setItem("openhands-telemetry-first-use", "true");

      await clearTelemetryData();

      expect(localStorage.getItem("openhands-telemetry-consent")).toBeNull();
      expect(getPendingCloudTelemetryConsent()).toBeNull();
      expect(localStorage.getItem("openhands-telemetry-first-use")).toBeNull();
      expect(mockPosthog.reset).toHaveBeenCalledWith(true);
      expect(mockPosthog.opt_out_capturing).toHaveBeenCalled();
      await expect(getTelemetryDistinctIdForConsentSync()).resolves.toBe(
        "user-a",
      );
    });

    it("falls back to opting out if the SDK cannot reset", async () => {
      await setTelemetryConsent("granted");
      mockPosthog.reset.mockImplementationOnce(() => {
        throw new Error("reset failed");
      });
      vi.clearAllMocks();

      await expect(clearTelemetryData()).resolves.toBeUndefined();

      expect(mockPosthog.opt_out_capturing).toHaveBeenCalledOnce();
    });
  });

  describe("PostHog integration", () => {
    it("calls opt_in_capturing when consent is granted", async () => {
      await setTelemetryConsent("granted");
      expect(mockPosthog.opt_in_capturing).toHaveBeenCalled();
    });

    it("calls opt_out_capturing when consent is denied", async () => {
      await setTelemetryConsent("denied");
      expect(mockPosthog.opt_out_capturing).toHaveBeenCalled();
    });
  });
});
