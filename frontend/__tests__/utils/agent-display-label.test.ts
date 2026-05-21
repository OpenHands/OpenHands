import { describe, it, expect } from "vitest";
import { resolveAgentChip } from "#/utils/agent-display-label";
import type { ACPProviderConfig } from "#/api/option-service/option.types";

const PROVIDERS: ACPProviderConfig[] = [
  {
    key: "claude-code",
    display_name: "Claude Code",
    default_command: ["npx", "-y", "@agentclientprotocol/claude-agent-acp"],
  },
  {
    key: "codex",
    display_name: "Codex",
    default_command: ["npx", "-y", "@openai/codex-acp"],
  },
];

describe("resolveAgentChip", () => {
  describe("OpenHands branch", () => {
    it("returns kind=openhands with prettified text and raw tooltip", () => {
      const chip = resolveAgentChip(
        "openhands",
        "anthropic/claude-sonnet-4-5-20250929",
      );
      expect(chip).toEqual({
        kind: "openhands",
        text: "Claude Sonnet 4.5",
        tooltip: "anthropic/claude-sonnet-4-5-20250929",
      });
    });

    it("treats undefined agent_kind as the OpenHands branch", () => {
      const chip = resolveAgentChip(undefined, "openai/gpt-4o");
      expect(chip?.kind).toBe("openhands");
      expect(chip?.text).toBe("GPT-4o");
    });

    it("returns null when no llm_model is set", () => {
      expect(resolveAgentChip("openhands", null)).toBeNull();
      expect(resolveAgentChip("openhands", undefined)).toBeNull();
      expect(resolveAgentChip(undefined, null)).toBeNull();
    });
  });

  describe("ACP branch", () => {
    it("uses provider brand as text when llm_model is missing", () => {
      const chip = resolveAgentChip(
        "acp",
        null,
        { acp_server: "claude-code" },
        PROVIDERS,
      );
      expect(chip).toEqual({
        kind: "acp",
        text: "Claude Code",
        tooltip: "Claude Code",
      });
    });

    it("uses prettified llm_model as text and composes a brand+model tooltip", () => {
      const chip = resolveAgentChip(
        "acp",
        "anthropic/claude-opus-4-1",
        { acp_server: "claude-code" },
        PROVIDERS,
      );
      expect(chip).toEqual({
        kind: "acp",
        text: "Claude Opus 4.1",
        tooltip: "Claude Code · anthropic/claude-opus-4-1",
      });
    });

    it("falls back to plain 'ACP' when the provider key is unknown", () => {
      const chip = resolveAgentChip(
        "acp",
        null,
        { acp_server: "unknown-thing" },
        PROVIDERS,
      );
      expect(chip).toEqual({
        kind: "acp",
        text: "ACP",
        tooltip: "ACP",
      });
    });

    it("falls back to 'ACP' when the provider registry hasn't loaded yet", () => {
      const chip = resolveAgentChip("acp", null, { acp_server: "claude-code" });
      expect(chip?.text).toBe("ACP");
      expect(chip?.tooltip).toBe("ACP");
    });
  });
});
