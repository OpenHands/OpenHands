import { beforeEach, describe, expect, it, vi } from "vitest";
import type { BashOutput } from "@openhands/typescript-client";
import AcpService from "#/api/acp-service/acp-service.api";

// Capture the command the service runs and control the BashOutput it sees.
const executeCommand = vi.hoisted(() => vi.fn());
vi.mock("@openhands/typescript-client/clients", () => ({
  BashClient: class {
    executeCommand = executeCommand;
  },
}));
vi.mock("#/api/agent-server-client-options", () => ({
  getAgentServerClientOptions: () => ({
    host: "http://localhost",
    workingDir: "/",
  }),
}));

function bashOutput(partial: Partial<BashOutput>): BashOutput {
  return {
    id: "1",
    timestamp: "2026-01-01T00:00:00Z",
    command_id: "c1",
    order: 0,
    exit_code: 0,
    stdout: null,
    stderr: null,
    kind: "BashOutput",
    ...partial,
  } as BashOutput;
}

beforeEach(() => vi.clearAllMocks());

describe("AcpService.getAuthStatus", () => {
  describe("claude-code (claude auth status --json)", () => {
    it("runs the right command and maps loggedIn:true → authenticated", async () => {
      executeCommand.mockResolvedValue(
        bashOutput({
          stdout: JSON.stringify({ loggedIn: true, authMethod: "claude.ai" }),
        }),
      );
      await expect(AcpService.getAuthStatus("claude-code")).resolves.toBe(
        "authenticated",
      );
      expect(executeCommand).toHaveBeenCalledWith(
        "claude auth status --json",
        undefined,
        expect.any(Number),
      );
    });

    it("maps loggedIn:false (even with a non-zero exit) → unauthenticated", async () => {
      executeCommand.mockResolvedValue(
        bashOutput({
          stdout: JSON.stringify({ loggedIn: false }),
          exit_code: 1,
        }),
      );
      await expect(AcpService.getAuthStatus("claude-code")).resolves.toBe(
        "unauthenticated",
      );
    });

    it("→ unknown when the CLI is missing (exit 127, no JSON on stdout)", async () => {
      // The "no available ACP process / CLI not installed" path.
      executeCommand.mockResolvedValue(
        bashOutput({
          exit_code: 127,
          stderr: "env: claude: No such file or directory",
        }),
      );
      await expect(AcpService.getAuthStatus("claude-code")).resolves.toBe(
        "unknown",
      );
    });
  });

  describe("codex (codex login status)", () => {
    it("→ authenticated even though the CLI writes to stderr", async () => {
      executeCommand.mockResolvedValue(
        bashOutput({ stderr: "Logged in using ChatGPT\n" }),
      );
      await expect(AcpService.getAuthStatus("codex")).resolves.toBe(
        "authenticated",
      );
    });

    it("→ unauthenticated on 'Not logged in'", async () => {
      executeCommand.mockResolvedValue(
        bashOutput({ stderr: "Not logged in\n" }),
      );
      await expect(AcpService.getAuthStatus("codex")).resolves.toBe(
        "unauthenticated",
      );
    });

    it("→ unknown when the CLI is missing", async () => {
      executeCommand.mockResolvedValue(
        bashOutput({ exit_code: 127, stderr: "codex: command not found" }),
      );
      await expect(AcpService.getAuthStatus("codex")).resolves.toBe("unknown");
    });
  });

  describe("gemini-cli (credentials file check)", () => {
    it("→ authenticated when the creds file is present", async () => {
      executeCommand.mockResolvedValue(bashOutput({ stdout: "present\n" }));
      await expect(AcpService.getAuthStatus("gemini-cli")).resolves.toBe(
        "authenticated",
      );
      // Probes the OAuth creds file, not a (nonexistent) gemini status command.
      expect(executeCommand.mock.calls[0][0]).toContain("oauth_creds.json");
    });

    it("→ unauthenticated when the creds file is absent", async () => {
      executeCommand.mockResolvedValue(bashOutput({ stdout: "absent\n" }));
      await expect(AcpService.getAuthStatus("gemini-cli")).resolves.toBe(
        "unauthenticated",
      );
    });
  });

  it("→ unknown for an unprobeable provider, without running any command", async () => {
    await expect(AcpService.getAuthStatus("openhands")).resolves.toBe(
      "unknown",
    );
    expect(executeCommand).not.toHaveBeenCalled();
  });
});
