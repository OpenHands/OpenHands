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

  it("masks secret-named query parameters even when the userinfo password contains a malformed percent escape", () => {
    // A lone `%` in the password is a valid `new URL()` parse but throws
    // inside `decodeURIComponent`. The throw previously skipped the
    // `searchParams` pass and silently leaked `?api_key=…` in the
    // surfaced error text. The password itself is still collected
    // because its raw form was already added before the decode attempt.
    const server: MCPServerConfig = {
      id: "shttp-0",
      type: "shttp",
      url: "https://user:pa%ssword123@mcp.example.com/mcp?api_key=querysecret456",
    };

    const redacted = redactMcpSecrets(
      "401 for url https://user:pa%ssword123@mcp.example.com/mcp?api_key=querysecret456",
      server,
    );

    expect(redacted).not.toContain("querysecret456");
    expect(redacted).not.toContain("pa%ssword123");
  });

  it("masks secret-named query parameters when the username contains a malformed percent escape", () => {
    // Symmetric to the password case: a malformed `%` in the username
    // would previously abort the URL parse helper and skip the
    // `searchParams` pass, leaking the query-string secret.
    const server: MCPServerConfig = {
      id: "shttp-0",
      type: "shttp",
      url: "https://us%er:urlpass123@mcp.example.com/mcp?api_key=querysecret456",
    };

    const redacted = redactMcpSecrets(
      "401 for url https://us%er:urlpass123@mcp.example.com/mcp?api_key=querysecret456",
      server,
    );

    expect(redacted).not.toContain("querysecret456");
    expect(redacted).not.toContain("us%er");
    expect(redacted).not.toContain("urlpass123");
  });
});
