import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";

const useTelemetryMock = vi.hoisted(() => vi.fn());

vi.mock("#/hooks/use-telemetry", () => ({
  useTelemetry: useTelemetryMock,
}));

import { TelemetryProvider } from "#/components/providers/telemetry-provider";
import * as telemetry from "#/services/telemetry";

const runtimeConfig = {
  apiKey: "phc_embedded",
  apiHost: "https://events.example.com",
  uiHost: "https://posthog.example.com",
};

function encodeHandoff(value: unknown): string {
  const encoded = btoa(
    encodeURIComponent(JSON.stringify(value)).replace(
      /%([0-9A-F]{2})/g,
      (_, hex: string) => String.fromCharCode(Number.parseInt(hex, 16)),
    ),
  );
  return encoded.replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}

describe("TelemetryProvider", () => {
  let configureBootstrapMock: ReturnType<typeof vi.spyOn>;
  let configureTelemetryMock: ReturnType<typeof vi.spyOn>;
  let initializeClientMock: ReturnType<typeof vi.spyOn>;
  let setWebsiteAttributionMock: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    configureBootstrapMock = vi
      .spyOn(telemetry, "configurePostHogBootstrap")
      .mockImplementation(() => undefined);
    configureTelemetryMock = vi
      .spyOn(telemetry, "configureTelemetry")
      .mockImplementation(() => undefined);
    initializeClientMock = vi
      .spyOn(telemetry, "initializePostHogClient")
      .mockResolvedValue(null);
    setWebsiteAttributionMock = vi
      .spyOn(telemetry, "setTelemetryWebsiteAttribution")
      .mockImplementation(() => undefined);
    useTelemetryMock.mockClear();
    window.location.hash = "";
    localStorage.clear();
    sessionStorage.clear();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("configures telemetry and bootstraps IDs from the URL", () => {
    window.location.hash = "distinct_id=user-123&session_id=session-456";

    render(
      <TelemetryProvider config={runtimeConfig}>
        <div data-testid="child" />
      </TelemetryProvider>,
    );

    expect(configureTelemetryMock).toHaveBeenCalledWith(runtimeConfig);
    expect(configureBootstrapMock).toHaveBeenCalledWith({
      distinctID: "user-123",
      sessionID: "session-456",
    });
    expect(window.location.hash).toBe("");
  });

  it("restores valid bootstrap IDs after OAuth and discards malformed data", () => {
    sessionStorage.setItem("posthog_bootstrap", "not-json");
    const view = render(
      <TelemetryProvider config={runtimeConfig}>
        <div />
      </TelemetryProvider>,
    );

    expect(configureBootstrapMock).toHaveBeenCalledWith(undefined);
    expect(sessionStorage.getItem("posthog_bootstrap")).toBeNull();

    view.unmount();
    configureBootstrapMock.mockClear();
    sessionStorage.setItem(
      "posthog_bootstrap",
      JSON.stringify({ distinctID: "user-123", sessionID: "session-456" }),
    );
    render(
      <TelemetryProvider config={runtimeConfig}>
        <div />
      </TelemetryProvider>,
    );
    expect(configureBootstrapMock).toHaveBeenCalledWith({
      distinctID: "user-123",
      sessionID: "session-456",
    });
  });

  it("consumes structured website handoff IDs and attribution from the URL", () => {
    window.location.hash = `oh_ph_handoff=${encodeHandoff({
      v: 1,
      exp: Date.now() + 60_000,
      nonce: "nonce-structured",
      distinct_id: "website-anon-id",
      session_id: "website-session-id",
      attribution: {
        utm_source: "newsletter",
        utm_medium: "email",
        utm_campaign: "launch",
        landing_page_category: "home",
        cta_id: "hero-cloud",
        cta_surface: "homepage_hero",
        referring_domain_category: "search",
        full_url: "https://www.openhands.dev/?secret=value",
      },
    })}`;

    render(
      <TelemetryProvider config={runtimeConfig}>
        <div data-testid="child" />
      </TelemetryProvider>,
    );

    expect(configureBootstrapMock).toHaveBeenCalledWith({
      distinctID: "website-anon-id",
      sessionID: "website-session-id",
    });
    expect(setWebsiteAttributionMock).toHaveBeenCalledWith({
      utm_source: "newsletter",
      utm_medium: "email",
      utm_campaign: "launch",
      landing_page_category: "home",
      cta_id: "hero-cloud",
      cta_surface: "homepage_hero",
      referring_domain_category: "search",
    });
    expect(window.location.hash).toBe("");
  });

  it("drops malformed or expired handoffs and removes them from the URL", () => {
    window.location.hash = `oh_ph_handoff=${encodeHandoff({
      v: 1,
      exp: Date.now() - 1,
      nonce: "nonce-expired",
      distinct_id: "website-anon-id",
      session_id: "website-session-id",
    })}`;

    render(
      <TelemetryProvider config={runtimeConfig}>
        <div />
      </TelemetryProvider>,
    );

    expect(configureBootstrapMock).toHaveBeenCalledWith(undefined);
    expect(setWebsiteAttributionMock).toHaveBeenCalledWith(undefined);
    expect(window.location.hash).toBe("");
  });

  it("prevents replay of a consumed structured handoff URL", () => {
    const encoded = encodeHandoff({
      v: 1,
      exp: Date.now() + 60_000,
      nonce: "nonce-replay",
      distinct_id: "website-anon-id",
      session_id: "website-session-id",
    });

    window.location.hash = `oh_ph_handoff=${encoded}`;
    const firstView = render(
      <TelemetryProvider config={runtimeConfig}>
        <div />
      </TelemetryProvider>,
    );
    expect(configureBootstrapMock).toHaveBeenLastCalledWith({
      distinctID: "website-anon-id",
      sessionID: "website-session-id",
    });

    firstView.unmount();
    configureBootstrapMock.mockClear();
    setWebsiteAttributionMock.mockClear();
    window.location.hash = `oh_ph_handoff=${encoded}`;

    render(
      <TelemetryProvider config={runtimeConfig}>
        <div />
      </TelemetryProvider>,
    );

    expect(configureBootstrapMock).toHaveBeenCalledWith(undefined);
    expect(setWebsiteAttributionMock).toHaveBeenCalledWith(undefined);
  });

  it("mounts telemetry lifecycle when analytics are enabled", () => {
    render(
      <TelemetryProvider config={runtimeConfig}>
        <div data-testid="child" />
      </TelemetryProvider>,
    );

    expect(screen.getByTestId("child")).toBeInTheDocument();
    expect(useTelemetryMock).toHaveBeenCalledOnce();
  });

  it("keeps rendering children when eager initialization fails", async () => {
    initializeClientMock.mockRejectedValueOnce(new Error("unavailable"));

    render(
      <TelemetryProvider config={runtimeConfig}>
        <div data-testid="child" />
      </TelemetryProvider>,
    );

    expect(screen.getByTestId("child")).toBeInTheDocument();
    await waitFor(() => expect(initializeClientMock).toHaveBeenCalledOnce());
  });

  it("does not initialize telemetry lifecycle when analytics are disabled", () => {
    render(
      <TelemetryProvider config={false}>
        <div data-testid="child" />
      </TelemetryProvider>,
    );

    expect(screen.getByTestId("child")).toBeInTheDocument();
    expect(configureTelemetryMock).toHaveBeenCalledWith(false);
    expect(initializeClientMock).not.toHaveBeenCalled();
    expect(useTelemetryMock).not.toHaveBeenCalled();
  });
});
