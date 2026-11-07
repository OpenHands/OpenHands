import { describe, expect, it } from "vitest";

import {
  LAUNCH_CHILD_CONVERSATION_ACTION_KIND,
  LAUNCH_CHILD_CONVERSATION_CLIENT_TOOL,
  LAUNCH_CHILD_CONVERSATION_TOOL_NAME,
} from "#/api/launch-child-conversation-client-tool";

describe("launch_child_conversation client tool", () => {
  it("exports the semantic tool name and generated action kind", () => {
    expect(LAUNCH_CHILD_CONVERSATION_TOOL_NAME).toBe(
      "launch_child_conversation",
    );
    expect(LAUNCH_CHILD_CONVERSATION_ACTION_KIND).toBe(
      "ClientAction_launch_child_conversation",
    );
    expect(LAUNCH_CHILD_CONVERSATION_CLIENT_TOOL.name).toBe(
      LAUNCH_CHILD_CONVERSATION_TOOL_NAME,
    );
  });

  it("pins the validated parameter schema", () => {
    expect(LAUNCH_CHILD_CONVERSATION_CLIENT_TOOL.parameters).toMatchObject({
      type: "object",
      // Misspelled parameter names must fail server-side (the SDK builds a
      // pydantic model with extra="forbid") instead of launching a child with
      // silently dropped arguments.
      additionalProperties: false,
      properties: {
        target: { enum: ["local", "cloud"] },
        isolation: { enum: ["worktree", "shared"] },
        agent_profile: { type: "string" },
      },
      required: ["target", "task"],
    });
  });

  it("is annotated as a non-idempotent, outward-reaching tool", () => {
    expect(LAUNCH_CHILD_CONVERSATION_CLIENT_TOOL.annotations).toEqual({
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: false,
      openWorldHint: true,
    });
  });

  it("tells the agent the result arrives as a follow-up message", () => {
    expect(LAUNCH_CHILD_CONVERSATION_CLIENT_TOOL.description).toContain(
      "[child-conversation]",
    );
  });

  it("documents a concrete launch and how profile selection differs from sub-agents", () => {
    const description = LAUNCH_CHILD_CONVERSATION_CLIENT_TOOL.description;

    expect(description).toContain('target: "local"');
    expect(description).toContain('target: "cloud"');
    expect(description).toContain('agent_profile: "fast"');
    expect(description).toContain('repository: "octocat/hello-world"');
    expect(description).toContain('Omit "agent_profile"');
    expect(description).toContain("sub-agents");
    expect(description).toContain("secret allow-list");
    expect(description).toContain("not confirmed");
    expect(description).toContain("organization-only");
    expect(LAUNCH_CHILD_CONVERSATION_CLIENT_TOOL.parameters.required).toEqual([
      "target",
      "task",
    ]);
  });
});
