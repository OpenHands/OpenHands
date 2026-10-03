import { describe, expect, it } from "vitest";
import type { MCPServerConfig } from "#/types/mcp-server";
import { REDACTED_MCP_SECRET_VALUE } from "#/utils/mcp-config";
import { collectMcpSecretValues, redactMcpSecrets } from "#/utils/redact-mcp-secrets";

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

describe("collectMcpSecretValues, URL query-string secrets", () => {
  const server = (url: string): MCPServerConfig =>
    ({
      id: "s1",
      name: "s1",
      type: "sse",
      url,
    }) as unknown as MCPServerConfig;

  it("collects ?api_key= when the password is plain", () => {
    const s = server(
      "https://alice:plainpassword@mcp.example.com/sse?api_key=QUERYSECRET1234",
    );
    expect(collectMcpSecretValues(s)).toContain("QUERYSECRET1234");
  });

  it("collects ?api_key= when the password contains a bare percent sign", () => {
    // Regression for issue #16978: a bare '%' in the password makes
    // decodeURIComponent throw, which used to abort the whole
    // addUrlSecrets() pass and silently drop the query-string secret.
    const s = server(
      "https://alice:pa%ssword@mcp.example.com/sse?api_key=QUERYSECRET1234",
    );
    expect(collectMcpSecretValues(s)).toContain("QUERYSECRET1234");
  });

  it("collects ?api_key= when the username contains a bare percent sign", () => {
    const s = server(
      "https://al%ice:password@mcp.example.com/sse?api_key=QUERYSECRET1234",
    );
    expect(collectMcpSecretValues(s)).toContain("QUERYSECRET1234");
  });

  it("still collects the raw username and password when percent decoding fails", () => {
    const s = server("https://alice:pa%ssword@mcp.example.com/sse");
    const values = collectMcpSecretValues(s);
    expect(values).toContain("alice");
    expect(values).toContain("pa%ssword");
  });

  it("redacts the query-string secret end-to-end when the password has a bare '%'", () => {
    const s = server(
      "https://alice:pa%ssword@mcp.example.com/sse?api_key=QUERYSECRET1234",
    );
    const redacted = redactMcpSecrets(
      "401 for https://alice:pa%ssword@mcp.example.com/sse?api_key=QUERYSECRET1234",
      s,
    );
    expect(redacted).not.toContain("QUERYSECRET1234");
    expect(redacted).toContain(REDACTED_MCP_SECRET_VALUE);
  });

  it("collects nothing and does not throw for a genuinely unparseable URL", () => {
    const s = server("not a url");
    expect(collectMcpSecretValues(s)).toEqual([]);
  });
});
