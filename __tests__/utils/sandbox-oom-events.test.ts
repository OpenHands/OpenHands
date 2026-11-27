import { describe, expect, it } from "vitest";
import { detectSandboxOOM, isOOMText } from "#/utils/sandbox-oom-events";
import type { OpenHandsEvent } from "#/types/agent-server/core";

describe("sandbox-oom-events", () => {
  describe("isOOMText", () => {
    it("detects standard OOM signatures", () => {
      expect(isOOMText("Process was Killed")).toBe(true);
      expect(isOOMText("fatal error: OOMKilled by cgroup")).toBe(true);
      expect(isOOMText("Python runtime out of memory")).toBe(true);
      expect(
        isOOMText("kernel: cgroup out of memory: killed process 1234"),
      ).toBe(true);
    });

    it("returns false for unrelated text", () => {
      expect(isOOMText("Compilation finished successfully")).toBe(false);
      expect(isOOMText("SyntaxError: unexpected token")).toBe(false);
      expect(isOOMText("Command not found")).toBe(false);
    });
  });

  describe("detectSandboxOOM", () => {
    it("detects bash observation with exit_code 137", () => {
      const events: OpenHandsEvent[] = [
        {
          id: "act-1",
          timestamp: "2026-01-01T00:00:00Z",
          source: "agent",
          tool_name: "execute_bash",
          tool_call_id: "call-1",
          action: {
            kind: "ExecuteBashAction",
            command: "npm run build:heavy",
          },
        } as unknown as OpenHandsEvent,
        {
          id: "evt-1",
          timestamp: "2026-01-01T00:00:00Z",
          source: "environment",
          action_id: "act-1",
          observation: {
            kind: "ExecuteBashObservation",
            content: [{ type: "text", text: "Command interrupted" }],
            exit_code: 137,
            metadata: {},
          },
        } as unknown as OpenHandsEvent,
      ];

      const detected = detectSandboxOOM(events);
      expect(detected).toHaveLength(1);
      expect(detected[0].eventId).toBe("evt-1");
      expect(detected[0].command).toBe("npm run build:heavy");
    });

    it("detects bash observation with OOM killed text even if exit_code is generic", () => {
      const events: OpenHandsEvent[] = [
        {
          id: "evt-2",
          timestamp: "2026-01-01T00:00:01Z",
          source: "environment",
          action_id: "act-2",
          observation: {
            kind: "ExecuteBashObservation",
            content: [{ type: "text", text: "Killed\nError running tests" }],
            exit_code: 1,
            metadata: {},
          },
        } as unknown as OpenHandsEvent,
      ];

      const detected = detectSandboxOOM(events);
      expect(detected).toHaveLength(1);
      expect(detected[0].eventId).toBe("evt-2");
    });

    it("detects displayable ConversationErrorEvent carrying OOM code or detail", () => {
      const events: OpenHandsEvent[] = [
        {
          id: "evt-3",
          timestamp: "2026-01-01T00:00:02Z",
          kind: "ConversationErrorEvent",
          code: "OOMKilled",
          detail: "Sandbox exceeded cgroup memory limit and terminated",
        } as unknown as OpenHandsEvent,
      ];

      const detected = detectSandboxOOM(events);
      expect(detected).toHaveLength(1);
      expect(detected[0].eventId).toBe("evt-3");
    });

    it("ignores normal events and successful commands", () => {
      const events: OpenHandsEvent[] = [
        {
          id: "evt-4",
          timestamp: "2026-01-01T00:00:03Z",
          source: "agent",
          kind: "ExecuteBashObservation",
          observation: {
            kind: "ExecuteBashObservation",
            content: [{ type: "text", text: "All tests passed" }],
            exit_code: 0,
            metadata: {},
          },
        } as unknown as OpenHandsEvent,
      ];

      const detected = detectSandboxOOM(events);
      expect(detected).toHaveLength(0);
    });
  });
});
