import { describe, expect, it } from "vitest";
import type { MCPServerConfig } from "#/types/mcp-server";
import { REDACTED_MCP_SECRET_VALUE } from "#/utils/mcp-config";
import { redactMcpSecrets } from "#/utils/redact-mcp-secrets";

describe("redactMcpSecrets", () => {
  it("masks configured secret values wherever they appear in the text", () => {
    const server: MCPServerConfig = {
      id: "stdio-0",
      type: "stdio",
      command: "npx",
      env: { API_TOKEN: "env-secret-value" },
    };

    const redacted = redactMcpSecrets(
      "spawn failed: API_TOKEN=env-secret-value was rejected",
      server,
    );

    expect(redacted).toBe(
      `spawn failed: API_TOKEN=${REDACTED_MCP_SECRET_VALUE} was rejected`,
    );
  });

  it("masks secrets embedded in the server URL (userinfo and secret-named query params)", () => {
    const server: MCPServerConfig = {
      id: "shttp-0",
      type: "shttp",
      url: "https://user:urlpass123@mcp.example.com/mcp?api_key=querysecret456",
    };

    const redacted = redactMcpSecrets(
      "401 for url https://user:urlpass123@mcp.example.com/mcp?api_key=querysecret456",
      server,
    );

    expect(redacted).not.toContain("urlpass123");
    expect(redacted).not.toContain("querysecret456");
  });

  it("masks well-known token shapes even without a server config", () => {
    const redacted = redactMcpSecrets(
      "rejected github_pat_ABCDEFGHIJKLMNOPQRSTUV and xoxb-1234-abcd " +
        "with header Bearer sometoken12345",
    );

    expect(redacted).not.toContain("github_pat_ABCDEFGHIJKLMNOPQRSTUV");
    expect(redacted).not.toContain("xoxb-1234-abcd");
    expect(redacted).not.toContain("sometoken12345");
    expect(redacted).toContain(REDACTED_MCP_SECRET_VALUE);
  });

  it("does not treat the redaction placeholder or very short values as secrets", () => {
    const server: MCPServerConfig = {
      id: "stdio-0",
      type: "stdio",
      command: "npx",
      env: { UNCHANGED: REDACTED_MCP_SECRET_VALUE, REGION: "eu" },
    };

    // "eu" appears inside ordinary words; masking it would mangle the text.
    const text = "could not resolve eu endpoint";

    expect(redactMcpSecrets(text, server)).toBe(text);
  });
});

it("still redacts query-string secrets when the URL userinfo contains a lone percent", () => {
  // issue #16978: a lone '%' in the password makes decodeURIComponent throw
  // URIError, which the whole addUrlSecrets block catches and aborts —
  // leaving the query-string secrets unredacted.
  const server: MCPServerConfig = {
    id: "shttp-0",
    type: "shttp",
    url: "https://user:pa%ssword@mcp.example.com/mcp?api_key=querysecret789",
  };

  const redacted = redactMcpSecrets(
    "401 for url https://user:pa%ssword@mcp.example.com/mcp?api_key=querysecret789",
    server,
  );

  expect(redacted).not.toContain("pa%ssword");
  expect(redacted).not.toContain("querysecret789");
});
