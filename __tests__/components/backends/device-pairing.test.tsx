import React from "react";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import { server } from "#/mocks/node";
import {
  __resetActiveStoreForTests,
  getActiveBackend,
  getRegisteredBackends,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import {
  buildDevicePairingLink,
  getCanvasHostHeaders,
  isUnencryptedNetworkUrl,
} from "#/api/mars/mars-device-pairing";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { DevicePairingButton } from "#/components/features/backends/managed-agents/device-pairing";
import PairDevice from "#/routes/pair-device";

vi.mock("#/services/telemetry", () => ({
  setTelemetryCloudContext: vi.fn(),
  setTelemetryIdentity: vi.fn(),
}));

beforeEach(() => {
  vi.stubEnv("VITE_SESSION_API_KEY", "");
  window.localStorage.clear();
  window.sessionStorage.clear();
  __resetActiveStoreForTests();
  setRegisteredBackends([]);
  window.history.replaceState({}, "", "/");
});

afterEach(() => {
  vi.unstubAllEnvs();
  window.history.replaceState({}, "", "/");
});

describe("device pairing", () => {
  it("creates once in StrictMode, keeps the secret in the fragment, and revokes on close", async () => {
    setRegisteredBackends([
      {
        id: "stale-host",
        name: "Old host key",
        kind: "local",
        host: window.location.origin,
        apiKey: "stale-host-key",
      },
      {
        id: "current-host",
        name: "Current host key",
        kind: "local",
        host: window.location.origin,
        apiKey: "current-host-key",
      },
    ]);
    setActiveSelection({ backendId: "current-host" });
    const created = vi.fn();
    const cancelled = vi.fn();
    server.use(
      http.post("*/mars/pairing/create", ({ request }) => {
        created(request.headers.get("X-Session-API-Key"));
        return HttpResponse.json({
          secret: "one-time-proof",
          expiresAt: Date.now() + 300_000,
        });
      }),
      http.post("*/mars/pairing/cancel", async ({ request }) => {
        cancelled(await request.json());
        return HttpResponse.json({});
      }),
    );
    const user = userEvent.setup();
    renderWithProviders(
      <React.StrictMode>
        <DevicePairingButton />
      </React.StrictMode>,
    );

    await user.click(
      screen.getByRole("button", { name: "BACKEND$PAIR_DEVICE" }),
    );
    const field = await screen.findByRole("textbox", {
      name: "BACKEND$PAIR_DEVICE_LINK",
    });
    const link = new URL((field as HTMLInputElement).value);
    expect(link.pathname).toBe("/pair");
    expect(link.search).toBe("");
    expect(link.hash).toContain("one-time-proof");
    expect(created).toHaveBeenCalledExactlyOnceWith("current-host-key");

    await user.clear(
      screen.getByRole("textbox", { name: "BACKEND$PAIR_DEVICE_ADDRESS" }),
    );
    await user.type(
      screen.getByRole("textbox", { name: "BACKEND$PAIR_DEVICE_ADDRESS" }),
      "http://studio.local:8000",
    );
    expect(screen.getByRole("alert")).toHaveTextContent(
      "BACKEND$UNENCRYPTED_CONNECTION",
    );
    expect(
      new URL(
        (
          screen.getByRole("textbox", {
            name: "BACKEND$PAIR_DEVICE_LINK",
          }) as HTMLInputElement
        ).value,
      ).host,
    ).toBe("studio.local:8000");
    await user.click(screen.getByRole("button", { name: "BUTTON$CLOSE" }));
    await waitFor(() =>
      expect(cancelled).toHaveBeenCalledWith({ secret: "one-time-proof" }),
    );
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("pairs without a typed key and preserves the selected DigitalOcean backend and conversation", async () => {
    const managed = {
      id: "do",
      name: "Team sandbox",
      kind: "local" as const,
      host: `${window.location.origin}/mars/sessions/s1`,
      apiKey: "",
      marsSessionId: "s1",
    };
    setRegisteredBackends([managed]);
    setActiveSelection({ backendId: managed.id });
    const redeem = vi.fn();
    server.use(
      http.post("*/mars/pairing/redeem", async ({ request }) => {
        redeem(await request.json());
        return HttpResponse.json({ sessionApiKey: "paired-host-key" });
      }),
    );
    window.history.replaceState({}, "", "/pair#canvas-pair=proof");
    renderWithProviders(
      <ActiveBackendProvider>
        <PairDevice />
      </ActiveBackendProvider>,
    );
    await waitFor(() => expect(window.location.hash).toBe(""));
    await userEvent.click(
      screen.getByRole("button", { name: "BACKEND$PAIR_DEVICE_CONNECT" }),
    );

    await waitFor(() => expect(getRegisteredBackends()).toHaveLength(2));
    expect(redeem).toHaveBeenCalledWith({ secret: "proof" });
    expect(getActiveBackend().backend).toEqual(managed);
    expect(getCanvasHostHeaders()).toEqual({
      "X-Session-API-Key": "paired-host-key",
    });
    __resetActiveStoreForTests();
    expect(getCanvasHostHeaders()).toEqual({
      "X-Session-API-Key": "paired-host-key",
    });
  });

  it("leaves the registry unchanged when a link is rejected", async () => {
    const local = {
      id: "local",
      name: "My Canvas",
      kind: "local" as const,
      host: window.location.origin,
      apiKey: "old-key",
    };
    setRegisteredBackends([local]);
    server.use(
      http.post(
        "*/mars/pairing/redeem",
        () => new HttpResponse(null, { status: 401 }),
      ),
    );
    window.history.replaceState({}, "", "/pair#canvas-pair=rejected-proof");
    renderWithProviders(
      <ActiveBackendProvider>
        <PairDevice />
      </ActiveBackendProvider>,
    );
    await userEvent.click(
      screen.getByRole("button", { name: "BACKEND$PAIR_DEVICE_CONNECT" }),
    );
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "BACKEND$PAIR_DEVICE_FAILED",
    );
    expect(getRegisteredBackends()).toEqual([local]);
  });

  it.each([
    ["http://localhost:8000", false],
    ["http://canvas.localhost:8000", false],
    ["http://127.0.0.2:8000", false],
    ["http://[::1]:8000", false],
    ["http://studio.local:8000", true],
    ["http://192.168.1.2:8000", true],
    ["https://studio.local", false],
  ])("warns only for unencrypted network connections: %s", (url, expected) => {
    expect(isUnencryptedNetworkUrl(url)).toBe(expected);
  });

  it("rejects unsafe pairing destinations and never borrows an unrelated backend's key", () => {
    setRegisteredBackends([
      {
        id: "remote",
        name: "Other host",
        kind: "local",
        host: "https://remote.example",
        apiKey: "remote-key",
      },
    ]);
    expect(getCanvasHostHeaders()).toEqual({});
    expect(() =>
      buildDevicePairingLink("javascript:alert(1)", "proof"),
    ).toThrow();
    expect(() =>
      buildDevicePairingLink("https://user:password@host", "proof"),
    ).toThrow();
  });
});
