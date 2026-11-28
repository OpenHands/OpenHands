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

  it("masks frontier model API keys (OpenAI, Anthropic, Gemini, Hugging Face)", () => {
    const errorText =
      "Failed to initialize with OPENAI_API_KEY=sk-proj-1234567890abcdefghijklmnopqrstuvwxyz12345 " +
      "or legacy sk-1234567890abcdefghijklmnopqrstuvwxyz12345. " +
      "ANTHROPIC_API_KEY=sk-ant-api03-abcdefghijklmnopqrstuvwxyz12345, " +
      "GEMINI_API_KEY=AIzaSyD-1234567890abcdefghijklmnopqrstuv, " +
      "HUGGINGFACE_TOKEN=hf_abcdefghijklmnopqrstuvwxyz12345678.";

    const redacted = redactMcpSecrets(errorText);

    expect(redacted).not.toContain(
      "sk-proj-1234567890abcdefghijklmnopqrstuvwxyz12345",
    );
    expect(redacted).not.toContain(
      "sk-1234567890abcdefghijklmnopqrstuvwxyz12345",
    );
    expect(redacted).not.toContain(
      "sk-ant-api03-abcdefghijklmnopqrstuvwxyz12345",
    );
    expect(redacted).not.toContain("AIzaSyD-1234567890abcdefghijklmnopqrstuv");
    expect(redacted).not.toContain("hf_abcdefghijklmnopqrstuvwxyz12345678");

    expect(redacted).toBe(
      `Failed to initialize with OPENAI_API_KEY=${REDACTED_MCP_SECRET_VALUE} ` +
        `or legacy ${REDACTED_MCP_SECRET_VALUE}. ` +
        `ANTHROPIC_API_KEY=${REDACTED_MCP_SECRET_VALUE}, ` +
        `GEMINI_API_KEY=${REDACTED_MCP_SECRET_VALUE}, ` +
        `HUGGINGFACE_TOKEN=${REDACTED_MCP_SECRET_VALUE}.`,
    );
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
