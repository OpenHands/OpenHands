import { describe, it, expect } from "vitest";
import {
  extractACPExecuteCommand,
  extractACPExecuteOutput,
} from "#/utils/acp-terminal";
import type { ACPToolCallEvent } from "#/types/agent-server/core/events/acp-tool-call-event";

describe("acp-terminal utils", () => {
  const createBaseEvent = (
    overrides: Partial<ACPToolCallEvent> = {},
  ): ACPToolCallEvent => ({
    id: "evt-1",
    timestamp: new Date().toISOString(),
    source: "agent",
    kind: "ACPToolCallEvent",
    tool_call_id: "call-1",
    title: "Run command",
    status: "completed",
    tool_kind: "execute",
    raw_input: null,
    raw_output: null,
    content: null,
    is_error: false,
    ...overrides,
  });

  describe("extractACPExecuteCommand", () => {
    it("extracts command from raw_input object with command field", () => {
      const event = createBaseEvent({
        raw_input: { command: "npm test -- --run" },
      });
      expect(extractACPExecuteCommand(event)).toBe("npm test -- --run");
    });

    it("extracts command from raw_input string", () => {
      const event = createBaseEvent({
        raw_input: "cargo check",
      });
      expect(extractACPExecuteCommand(event)).toBe("cargo check");
    });

    it("strips redundant prefixes from title when raw_input is empty", () => {
      const event = createBaseEvent({
        raw_input: null,
        title: "Bash git status",
      });
      expect(extractACPExecuteCommand(event)).toBe("git status");
    });

    it("falls back to event.title when prefix does not match", () => {
      const event = createBaseEvent({
        raw_input: null,
        title: "custom-cli --version",
      });
      expect(extractACPExecuteCommand(event)).toBe("custom-cli --version");
    });
  });

  describe("extractACPExecuteOutput", () => {
    it("extracts output from text content blocks", () => {
      const event = createBaseEvent({
        content: [
          { type: "text", text: "line 1" },
          { type: "text", text: "line 2" },
        ],
      });
      expect(extractACPExecuteOutput(event)).toBe("line 1\nline 2");
    });

    it("extracts output from wrapped content blocks with resource", () => {
      const event = createBaseEvent({
        content: [
          {
            type: "content",
            content: {
              type: "resource",
              resource: { text: "resource output text" },
            },
          },
        ],
      });
      expect(extractACPExecuteOutput(event)).toBe("resource output text");
    });

    it("falls back to raw_output when resource text is whitespace only", () => {
      const event = createBaseEvent({
        content: [
          {
            type: "content",
            content: {
              type: "resource",
              resource: { text: "   \n\t  " },
            },
          },
        ],
        raw_output: "Fallback raw output",
      });
      expect(extractACPExecuteOutput(event)).toBe("Fallback raw output");
    });

    it("falls back to raw_output when content has no text blocks", () => {
      const event = createBaseEvent({
        content: null,
        raw_output: "Done in 1.2s",
      });
      expect(extractACPExecuteOutput(event)).toBe("Done in 1.2s");
    });

    it("returns null when neither content nor raw_output is present on success", () => {
      const event = createBaseEvent({
        content: null,
        raw_output: null,
        is_error: false,
        status: "completed",
      });
      expect(extractACPExecuteOutput(event)).toBeNull();
    });

    it("returns error message when call failed without specific output", () => {
      const event = createBaseEvent({
        content: null,
        raw_output: null,
        is_error: true,
        status: "failed",
      });
      expect(extractACPExecuteOutput(event)).toBe("Command failed");
    });
  });
});
