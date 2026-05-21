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
  {
    key: "gemini-cli",
    display_name: "Gemini CLI",
    default_command: ["npx", "-y", "@google/gemini-cli-acp"],
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

  describe("ACP branch — known providers map to brand kinds", () => {
    it.each([
      ["claude-code", "acp-claude-code", "Claude Code"],
      ["codex", "acp-codex", "Codex"],
      ["gemini-cli", "acp-gemini-cli", "Gemini CLI"],
    ])(
      "provider key %s → kind %s, brand text %s (no model)",
      (providerKey, expectedKind, expectedText) => {
        const chip = resolveAgentChip(
          "acp",
          null,
          { acp_server: providerKey },
          PROVIDERS,
        );
        expect(chip).toEqual({
          kind: expectedKind,
          text: expectedText,
          tooltip: expectedText,
        });
      },
    );

    it("uses prettified llm_model as text with brand+model tooltip", () => {
      const chip = resolveAgentChip(
        "acp",
        "anthropic/claude-opus-4-1",
        { acp_server: "claude-code" },
        PROVIDERS,
      );
      expect(chip).toEqual({
        kind: "acp-claude-code",
        text: "Claude Opus 4.1",
        tooltip: "Claude Code · anthropic/claude-opus-4-1",
      });
    });

    it("falls back to acp-generic + 'ACP' when the provider key is unknown", () => {
      const chip = resolveAgentChip(
        "acp",
        null,
        { acp_server: "some-custom-thing" },
        PROVIDERS,
      );
      expect(chip).toEqual({
        kind: "acp-generic",
        text: "ACP",
        tooltip: "ACP",
      });
    });

    it("uses the brand icon from acp_server even when the provider registry hasn't loaded yet (text falls back to 'ACP')", () => {
      const chip = resolveAgentChip("acp", null, { acp_server: "claude-code" });
      // The acp_server key is canonical, so we can still pick the right brand
      // mark; the display_name is unavailable without the registry so the text
      // falls back to the generic "ACP" label.
      expect(chip?.kind).toBe("acp-claude-code");
      expect(chip?.text).toBe("ACP");
    });
  });
});
