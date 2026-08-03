import { describe, expect, it } from "vitest";
import { buildSlashCommandOutputScopeId } from "#/utils/slash-command-output-scope";

describe("buildSlashCommandOutputScopeId", () => {
  it("isolates equal conversation IDs across backends and organizations", () => {
    const local = buildSlashCommandOutputScopeId({
      backendId: "local-a",
      orgId: null,
      conversationId: "conversation-1",
    });
    const otherBackend = buildSlashCommandOutputScopeId({
      backendId: "local-b",
      orgId: null,
      conversationId: "conversation-1",
    });
    const otherOrganization = buildSlashCommandOutputScopeId({
      backendId: "cloud-a",
      orgId: "org-b",
      conversationId: "conversation-1",
    });

    expect(new Set([local, otherBackend, otherOrganization])).toHaveLength(3);
  });

  it("isolates the home output of each active selection", () => {
    expect(
      buildSlashCommandOutputScopeId({ backendId: "a", orgId: null }),
    ).not.toBe(
      buildSlashCommandOutputScopeId({ backendId: "b", orgId: null }),
    );
  });
});
