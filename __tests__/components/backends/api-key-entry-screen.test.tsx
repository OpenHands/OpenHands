import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { BACKENDS_STORAGE_KEY } from "#/api/backend-registry/storage";
import { __resetActiveStoreForTests } from "#/api/backend-registry/active-store";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import ApiKeyEntryScreen from "#/components/features/backends/api-key-entry-screen";

// ── Mocks ────────────────────────────────────────────────────────────

const getSettingsMock = vi.fn();

vi.mock("@openhands/typescript-client/clients", () => ({
  SettingsClient: vi.fn(function SettingsClientMock() {
    return { getSettings: getSettingsMock };
  }),
  // ServerClient needed by useBackendsHealth (imported transitively)
  ServerClient: vi.fn(function ServerClientMock() {
    return { getServerInfo: vi.fn().mockResolvedValue({ version: "1.28.0" }) };
  }),
}));

// Stub cloud org service used by ActiveBackendProvider
vi.mock("#/api/cloud/organization-service.api", () => ({
  getCurrentCloudApiKey: vi.fn().mockResolvedValue({
    orgId: null,
    isLegacyKey: true,
  }),
}));

// Capture reload calls without crashing jsdom
const reloadMock = vi.fn();

// ── Helpers ──────────────────────────────────────────────────────────

function renderScreen() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <ActiveBackendProvider>
        <ApiKeyEntryScreen />
      </ActiveBackendProvider>
    </QueryClientProvider>,
  );
}

/** Fill the required fields: name and api key. */
async function fillRequiredFields(
  user: ReturnType<typeof userEvent.setup>,
  { name = "My Server", apiKey = "some-key" } = {},
) {
  await user.type(screen.getByTestId("api-key-entry-name"), name);
  await user.type(screen.getByTestId("api-key-entry-api-key"), apiKey);
}

// ── Setup / teardown ─────────────────────────────────────────────────

const ORIGINAL_LOCATION = window.location;

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  getSettingsMock.mockReset();

  // Replace window.location with a spy-able version
  Object.defineProperty(window, "location", {
    configurable: true,
    value: {
      ...ORIGINAL_LOCATION,
      origin: "http://localhost:8000",
      hostname: "localhost",
      reload: reloadMock,
    },
  });
  reloadMock.mockReset();
});

afterEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  vi.unstubAllEnvs();
  Object.defineProperty(window, "location", {
    configurable: true,
    value: ORIGINAL_LOCATION,
  });
});

// ── Tests ────────────────────────────────────────────────────────────

describe("ApiKeyEntryScreen", () => {
  // @spec — UI: name + host (disabled) + api key + connect
  it("renders name, host (disabled), api key, and connect button", () => {
    renderScreen();

    // Name field
    const nameInput = screen.getByTestId("api-key-entry-name");
    expect(nameInput).toBeInTheDocument();
    expect(nameInput).toHaveValue("");

    // Host field — pre-filled from window.location.origin, disabled
    const hostInput = screen.getByTestId("api-key-entry-host");
    expect(hostInput).toBeInTheDocument();
    expect(hostInput).toBeDisabled();
    expect(hostInput).toHaveValue("http://localhost:8000");

    // API key field
    expect(screen.getByTestId("api-key-entry-api-key")).toBeInTheDocument();

    // Connect button
    expect(screen.getByTestId("api-key-entry-submit")).toBeInTheDocument();
  });

  // @spec — API key field always starts empty (stale key wipe)
  it("starts with an empty api key even when localStorage has a stale key", () => {
    // Seed the backend registry with the stale key
    window.localStorage.setItem(
      BACKENDS_STORAGE_KEY,
      JSON.stringify([
        {
          id: "default-local",
          name: "Local",
          host: "http://localhost:8000",
          apiKey: "old-stale-key-from-previous-session",
          kind: "local",
        },
      ]),
    );

    renderScreen();

    expect(screen.getByTestId("api-key-entry-api-key")).toHaveValue("");
  });

  // @spec — Connect button requires both name and api key
  it("disables Connect when name or api key is empty", async () => {
    renderScreen();
    const user = userEvent.setup();
    const submit = screen.getByTestId("api-key-entry-submit");

    // Both empty
    expect(submit).toBeDisabled();

    // Only api key filled → still disabled
    await user.type(screen.getByTestId("api-key-entry-api-key"), "key");
    expect(submit).toBeDisabled();

    // Name also filled → enabled
    await user.type(screen.getByTestId("api-key-entry-name"), "Server");
    expect(submit).not.toBeDisabled();
  });

  // @spec — Valid key: validates against GET /api/settings, persists, reloads
  it("validates the key before persisting and reloading", async () => {
    getSettingsMock.mockResolvedValueOnce({ llm_model: "test" });

    renderScreen();
    const user = userEvent.setup();

    await fillRequiredFields(user, { apiKey: "correct-key" });
    await user.click(screen.getByTestId("api-key-entry-submit"));

    await waitFor(() => {
      expect(getSettingsMock).toHaveBeenCalledTimes(1);
    });

    // Key persisted to backend registry storage
    const stored = JSON.parse(
      window.localStorage.getItem(BACKENDS_STORAGE_KEY) ?? "[]",
    );
    expect(stored[0].apiKey).toBe("correct-key");

    // Page reloaded
    expect(reloadMock).toHaveBeenCalled();
  });

  // @spec — 401 shows "Invalid API key", does NOT persist or reload
  it("shows 'Invalid API key' when the key is rejected with 401", async () => {
    getSettingsMock.mockRejectedValueOnce(
      Object.assign(new Error("Unauthorized"), {
        name: "HttpError",
        status: 401,
      }),
    );

    renderScreen();
    const user = userEvent.setup();

    await fillRequiredFields(user, { apiKey: "wrong-key" });
    await user.click(screen.getByTestId("api-key-entry-submit"));

    // Error status appears with "Invalid" text
    await waitFor(() => {
      expect(screen.getByTestId("api-key-entry-status")).toBeInTheDocument();
    });
    expect(screen.getByTestId("api-key-entry-status")).toHaveClass(
      "text-red-400",
    );
    expect(screen.getByTestId("api-key-entry-status").textContent).toContain(
      "AUTH$INVALID_KEY",
    );

    // Rejected key NOT persisted.
    expect(window.localStorage.getItem(BACKENDS_STORAGE_KEY)).not.toContain(
      "wrong-key",
    );

    // Page NOT reloaded
    expect(reloadMock).not.toHaveBeenCalled();
  });

  // @spec — Non-401 errors show the actual error message, not "Invalid key"
  it("shows 'Connection failed' with detail for non-auth errors (e.g. 500)", async () => {
    getSettingsMock.mockRejectedValueOnce(
      Object.assign(new Error("HTTP 500: Internal Server Error"), {
        name: "HttpError",
        status: 500,
      }),
    );

    renderScreen();
    const user = userEvent.setup();

    await fillRequiredFields(user, { apiKey: "correct-key" });
    await user.click(screen.getByTestId("api-key-entry-submit"));

    await waitFor(() => {
      expect(screen.getByTestId("api-key-entry-status")).toBeInTheDocument();
    });

    const statusText =
      screen.getByTestId("api-key-entry-status").textContent ?? "";
    // Shows "Connection failed" prefix, NOT "Invalid API key"
    expect(statusText).toContain("AUTH$CONNECTION_FAILED");
    expect(statusText).toContain("500");
    expect(statusText).not.toContain("AUTH$INVALID_KEY");

    // Rejected key NOT persisted.
    expect(window.localStorage.getItem(BACKENDS_STORAGE_KEY)).not.toContain(
      "correct-key",
    );

    expect(reloadMock).not.toHaveBeenCalled();
  });

  // @spec — Retry flow: wrong key → error → correct key → success
  it("allows retry after a failed attempt", async () => {
    getSettingsMock
      .mockRejectedValueOnce(
        Object.assign(new Error("Unauthorized"), {
          name: "HttpError",
          status: 401,
        }),
      )
      .mockResolvedValueOnce({ llm_model: "test" });

    renderScreen();
    const user = userEvent.setup();
    const apiKeyInput = screen.getByTestId("api-key-entry-api-key");

    // First attempt — wrong key
    await fillRequiredFields(user, { apiKey: "wrong-key" });
    await user.click(screen.getByTestId("api-key-entry-submit"));
    await waitFor(() => {
      expect(screen.getByTestId("api-key-entry-status")).toHaveClass(
        "text-red-400",
      );
    });

    // Retry — correct key
    await user.clear(apiKeyInput);
    await user.type(apiKeyInput, "correct-key");
    await user.click(screen.getByTestId("api-key-entry-submit"));

    await waitFor(() => {
      expect(reloadMock).toHaveBeenCalled();
    });

    const stored = JSON.parse(
      window.localStorage.getItem(BACKENDS_STORAGE_KEY) ?? "[]",
    );
    expect(stored[0].apiKey).toBe("correct-key");
  });

  // @spec — Stale key in localStorage does not contaminate the new key
  it("persists only the freshly-entered key, not the stale one", async () => {
    getSettingsMock.mockResolvedValueOnce({ llm_model: "test" });

    renderScreen();
    const user = userEvent.setup();

    // API key field is empty — stale key not visible
    const apiKeyInput = screen.getByTestId("api-key-entry-api-key");
    expect(apiKeyInput).toHaveValue("");

    // Enter fresh key
    await fillRequiredFields(user, { apiKey: "fresh-key-BBBB" });
    await user.click(screen.getByTestId("api-key-entry-submit"));

    await waitFor(() => {
      expect(reloadMock).toHaveBeenCalled();
    });

    // Stored key is the NEW one, not old + new concatenated
    const stored = JSON.parse(
      window.localStorage.getItem(BACKENDS_STORAGE_KEY) ?? "[]",
    );
    expect(stored[0].apiKey).toBe("fresh-key-BBBB");
  });
});
