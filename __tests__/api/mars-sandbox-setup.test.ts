import { afterEach, describe, expect, it, vi } from "vitest";

const executeCommand = vi.hoisted(() => vi.fn());
const getSettings = vi.hoisted(() => vi.fn());
const updateSettings = vi.hoisted(() => vi.fn(async () => ({})));
const waitForMarsAgentServer = vi.hoisted(() => vi.fn(async () => "1.53.0"));

vi.mock("@openhands/typescript-client/clients", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@openhands/typescript-client/clients")
  >()),
  BashClient: vi.fn(function BashClient() {
    return { executeCommand };
  }),
  SettingsClient: vi.fn(function SettingsClient() {
    return { getSettings, updateSettings };
  }),
}));
vi.mock("#/api/mars/mars-tunnel-backend", async (importOriginal) => ({
  ...(await importOriginal<typeof import("#/api/mars/mars-tunnel-backend")>()),
  waitForMarsAgentServer,
}));

import {
  ENSURE_SECRET_KEY_SCRIPT,
  MARS_SANDBOX_TOOLS,
  prepareMarsSandbox,
} from "#/api/mars/mars-sandbox-setup";

const HOST = "http://127.0.0.1:51000";

afterEach(() => {
  vi.useRealTimers();
  vi.clearAllMocks();
});

describe("prepareMarsSandbox", () => {
  it("waits for the restarted server and moves the terminal off tmux", async () => {
    // Arrange
    vi.useFakeTimers();
    executeCommand.mockResolvedValue({ stdout: "restarting\n" });
    getSettings.mockResolvedValue({ agent_settings: { tools: null } });

    // Act
    const done = prepareMarsSandbox(HOST, "sess_1");
    await vi.runAllTimersAsync();
    await done;

    // Assert
    expect(executeCommand).toHaveBeenCalledWith(
      expect.objectContaining({ command: ENSURE_SECRET_KEY_SCRIPT }),
    );
    expect(waitForMarsAgentServer).toHaveBeenCalledWith(HOST, {
      sessionId: "sess_1",
    });
    expect(updateSettings).toHaveBeenCalledWith({
      agent_settings_diff: { tools: MARS_SANDBOX_TOOLS },
    });
  });

  it("leaves a ready sandbox and an explicit tool list alone", async () => {
    executeCommand.mockResolvedValue({ stdout: "ready\n" });
    getSettings.mockResolvedValue({
      agent_settings: { tools: [{ name: "terminal", params: {} }] },
    });

    await prepareMarsSandbox(HOST, "sess_1");

    expect(waitForMarsAgentServer).not.toHaveBeenCalled();
    expect(updateSettings).not.toHaveBeenCalled();
  });

  it("never blocks connecting when the sandbox cannot be prepared", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    executeCommand.mockRejectedValue(new Error("bash endpoint gone"));

    await expect(prepareMarsSandbox(HOST, "sess_1")).resolves.toBeUndefined();
    expect(updateSettings).not.toHaveBeenCalled();
  });
});
