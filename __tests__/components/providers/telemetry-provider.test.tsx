import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { TelemetryProvider } from "#/components/providers/telemetry-provider";
import * as telemetry from "#/services/telemetry";

const runtimeConfig = {
  apiKey: "phc_embedded",
  apiHost: "https://events.example.com",
  uiHost: "https://posthog.example.com",
};

describe("TelemetryProvider", () => {
  let configureBootstrapMock: ReturnType<typeof vi.spyOn>;
  let configureTelemetryMock: ReturnType<typeof vi.spyOn>;
  let initializeClientMock: ReturnType<typeof vi.spyOn>;

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
    window.location.hash = "";
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

  it("does not initialize when analytics are disabled", () => {
    render(
      <TelemetryProvider config={false}>
        <div data-testid="child" />
      </TelemetryProvider>,
    );

    expect(screen.getByTestId("child")).toBeInTheDocument();
    expect(configureTelemetryMock).toHaveBeenCalledWith(false);
    expect(initializeClientMock).not.toHaveBeenCalled();
  });
});
