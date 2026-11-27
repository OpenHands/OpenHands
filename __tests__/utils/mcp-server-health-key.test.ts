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

    expect(getMcpServerHealthKey(redactedLater)).toBe(
      getMcpServerHealthKey(GITHUB),
    );
  });

  it("distinguishes same-shaped servers by name", () => {
    // Duplicate installs of one catalog entry are stored as `github` and
    // `github_1`; their credentials (and health) are independent.
    const second: MCPServerConfig = { ...GITHUB, name: "github_1" };

    expect(getMcpServerHealthKey(second)).not.toBe(
      getMcpServerHealthKey(GITHUB),
    );
  });

  it("changes when the structural config changes", () => {
    const movedUrl: MCPServerConfig = {
      ...GITHUB,
      url: "https://other.example.com/mcp",
    };

    expect(getMcpServerHealthKey(movedUrl)).not.toBe(
      getMcpServerHealthKey(GITHUB),
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

    expect(getMcpServerHealthKey(rotatedToken)).toBe(
      getMcpServerHealthKey(stdio),
    );
    expect(getMcpServerHealthKey(extraEnvVar)).not.toBe(
      getMcpServerHealthKey(stdio),
    );
  });
});
