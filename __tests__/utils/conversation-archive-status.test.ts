import { describe, expect, it } from "vitest";
import {
  isEffectivelyArchivedConversation,
  isMissingSandboxStatus,
  isRuntimeUnavailableSandboxStatus,
} from "#/utils/conversation-archive-status";
import type { SandboxStatus } from "#/api/conversation-service/agent-server-conversation-service.types";

describe("conversation archive predicates", () => {
  it.each<[SandboxStatus | null | undefined, boolean]>([
    ["MISSING", true],
    ["ERROR", false],
    ["PAUSED", false],
    ["RUNNING", false],
    ["STARTING", false],
    [null, false],
    [undefined, false],
  ])("isMissingSandboxStatus(%s) === %s", (status, expected) => {
    expect(isMissingSandboxStatus(status)).toBe(expected);
  });

  it.each<[SandboxStatus | null | undefined, boolean]>([
    ["MISSING", true],
    ["ERROR", true],
    ["PAUSED", false],
    ["RUNNING", false],
    [null, false],
    [undefined, false],
  ])("isRuntimeUnavailableSandboxStatus(%s) === %s", (status, expected) => {
    expect(isRuntimeUnavailableSandboxStatus(status)).toBe(expected);
  });

  it("treats a missing runtime as effectively archived", () => {
    expect(isEffectivelyArchivedConversation("MISSING", false)).toBe(true);
  });

  it("treats an explicit archive as effectively archived", () => {
    expect(isEffectivelyArchivedConversation(null, true)).toBe(true);
    expect(isEffectivelyArchivedConversation("RUNNING", true)).toBe(true);
  });

  it("does not treat an errored runtime as archived on its own", () => {
    expect(isEffectivelyArchivedConversation("ERROR", false)).toBe(false);
  });

  it("does not treat a running runtime as archived on its own", () => {
    expect(isEffectivelyArchivedConversation("RUNNING", false)).toBe(false);
    expect(isEffectivelyArchivedConversation(null, false)).toBe(false);
  });
});
