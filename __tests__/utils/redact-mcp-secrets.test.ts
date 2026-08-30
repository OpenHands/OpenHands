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

  it("still collects query-string secrets when the URL password has a malformed percent escape (fixes #16978)", () => {
    // The userinfo "alice:pa%ssword" has a "%" that is not followed by two hex digits,
    // which makes decodeURIComponent throw URIError. The collection must not abort:
    // the raw password is still added by the addValue() call before the decode, and
    // the API-key query parameter on the same URL must also be picked up.
    const server: MCPServerConfig = {
      id: "shttp-1",
      type: "shttp",
      url: "https://alice:pa%ssword@mcp.example.com/mcp?api_key=querysecret456",
    };

    const redacted = redactMcpSecrets(
      "401 for https://alice:pa%ssword@mcp.example.com/mcp?api_key=querysecret456",
      server,
    );

    expect(redacted).not.toContain("querysecret456");
    // The raw "%ssword" survives too — the redactor only knows the literal text.
    expect(redacted).not.toContain("pa%ssword");
    expect(redacted).toContain(REDACTED_MCP_SECRET_VALUE);
  });
});
