import { describe, expect, it } from "vitest";
import { redactMcpSecrets } from "./redact-mcp-secrets";
import { REDACTED_MCP_SECRET_VALUE } from "./mcp-config";

describe("redactMcpSecrets", () => {
  it("redacts classic and fine-grained GitHub PATs", () => {
    const error = "Git clone failed with token ghp_0123456789abcdef0123456789abcdef";
    const redacted = redactMcpSecrets(error);
    expect(redacted).not.toContain("ghp_0123456789abcdef0123456789abcdef");
    expect(redacted).toContain(REDACTED_MCP_SECRET_VALUE);
  });

  it("redacts OpenAI API keys (both legacy sk- and project sk-proj- keys)", () => {
    const error = "OpenAI Error: API key sk-proj-abcdefghijklmnopqrstuvwxyz1234567890 exceeded quota";
    const redacted = redactMcpSecrets(error);
    expect(redacted).not.toContain("sk-proj-abcdefghijklmnopqrstuvwxyz1234567890");
    expect(redacted).toContain(REDACTED_MCP_SECRET_VALUE);
  });

  it("redacts Anthropic Claude API keys", () => {
    const error = "Anthropic Error: 401 Unauthorized for x-api-key sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890";
    const redacted = redactMcpSecrets(error);
    expect(redacted).not.toContain("sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890");
    expect(redacted).toContain(REDACTED_MCP_SECRET_VALUE);
  });

  it("redacts Google Gemini / Cloud API keys", () => {
    const error = "Request failed: https://generativelanguage.googleapis.com/v1beta/models?key=AIzaSyD-1234567890abcdefghijklmnopqrstuvw";
    const redacted = redactMcpSecrets(error);
    expect(redacted).not.toContain("AIzaSyD-1234567890abcdefghijklmnopqrstuvw");
    expect(redacted).toContain(REDACTED_MCP_SECRET_VALUE);
  });

  it("redacts Hugging Face API access tokens", () => {
    const error = "InferenceClient error with token hf_abcdefghijklmnopqrstuvwxyz1234567890";
    const redacted = redactMcpSecrets(error);
    expect(redacted).not.toContain("hf_abcdefghijklmnopqrstuvwxyz1234567890");
    expect(redacted).toContain(REDACTED_MCP_SECRET_VALUE);
  });
});
