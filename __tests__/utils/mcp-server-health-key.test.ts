import { describe, expect, it } from "vitest";
import type { MCPServerConfig } from "#/types/mcp-server";
import { REDACTED_MCP_SECRET_VALUE } from "#/utils/mcp-config";
import { getMcpServerHealthKey } from "#/utils/mcp-server-health-key";

const GITHUB: MCPServerConfig = {
  id: "shttp-0",
  type: "shttp",
  name: "github",
  url: "https://api.githubcopilot.com/mcp/",
  auth: { strategy: "api_key", value: "github_pat_real" },
};

const SCOPE_A = { backendId: "backend-a", connectionRevision: 0 };
const SCOPE_B = { backendId: "backend-b", connectionRevision: 0 };

describe("getMcpServerHealthKey", () => {
  it("is stable across positional ids and secret-value forms of the same server", () => {
    // The same stored server appears with plaintext at install time, the
    // redaction placeholder after a settings read, and a shifted positional
    // id after an unrelated delete — all must share one health entry.
    const redactedLater: MCPServerConfig = {
      ...GITHUB,
      id: "shttp-3",
      auth: { strategy: "api_key", value: REDACTED_MCP_SECRET_VALUE },
    };

    expect(getMcpServerHealthKey(SCOPE_A, redactedLater)).toBe(
      getMcpServerHealthKey(SCOPE_A, GITHUB),
    );
  });

  it("distinguishes same-shaped servers by name", () => {
    // Duplicate installs of one catalog entry are stored as `github` and
    // `github_1`; their credentials (and health) are independent.
    const second: MCPServerConfig = { ...GITHUB, name: "github_1" };

    expect(getMcpServerHealthKey(SCOPE_A, second)).not.toBe(
      getMcpServerHealthKey(SCOPE_A, GITHUB),
    );
  });

  it("changes when the structural config changes", () => {
    const movedUrl: MCPServerConfig = {
      ...GITHUB,
      url: "https://other.example.com/mcp",
    };

    expect(getMcpServerHealthKey(SCOPE_A, movedUrl)).not.toBe(
      getMcpServerHealthKey(SCOPE_A, GITHUB),
    );
  });

  it("tracks stdio env structure but not env values", () => {
    const stdio: MCPServerConfig = {
      id: "stdio-0",
      type: "stdio",
      name: "slack",
      command: "npx",
      args: ["-y", "@zencoderai/slack-mcp-server"],
      env: { SLACK_BOT_TOKEN: "xoxb-one" },
    };
    const rotatedToken: MCPServerConfig = {
      ...stdio,
      env: { SLACK_BOT_TOKEN: "xoxb-two" },
    };
    const extraEnvVar: MCPServerConfig = {
      ...stdio,
      env: { SLACK_BOT_TOKEN: "xoxb-one", SLACK_TEAM_ID: "T01" },
    };

    expect(getMcpServerHealthKey(SCOPE_A, rotatedToken)).toBe(
      getMcpServerHealthKey(SCOPE_A, stdio),
    );
    expect(getMcpServerHealthKey(SCOPE_A, extraEnvVar)).not.toBe(
      getMcpServerHealthKey(SCOPE_A, stdio),
    );
  });

  it("keys the same server independently under two backend scopes", () => {
    // The same catalog entry installed against backend A and backend B
    // carries two independent credentials and reachability verdicts; the
    // health map must not bleed one backend's state into the other.
    expect(getMcpServerHealthKey(SCOPE_A, GITHUB)).not.toBe(
      getMcpServerHealthKey(SCOPE_B, GITHUB),
    );
  });

  it("rotates the key when the backend's connection revision advances", () => {
    // A backend that re-handshakes with new credentials (host/api-key
    // rotation) must restart its MCP health from "unchecked" without an
    // explicit clearMcpServerHealth call.
    const before = getMcpServerHealthKey(
      { backendId: "backend-a", connectionRevision: 1 },
      GITHUB,
    );
    const after = getMcpServerHealthKey(
      { backendId: "backend-a", connectionRevision: 2 },
      GITHUB,
    );
    expect(before).not.toBe(after);
  });
});
