import { describe, expect, it } from "vitest";
import { getAgentProfileLlmDrift } from "#/components/features/settings/agent-profiles/agent-profile-llm-drift";
import { type AgentProfileSummary } from "#/api/agent-profiles-service/agent-profiles-service.api";

const openHandsProfile = (
  llmProfileRef: string | null,
): AgentProfileSummary => ({
  id: "id-oh",
  name: "default",
  agent_kind: "openhands",
  revision: 0,
  llm_profile_ref: llmProfileRef,
  mcp_server_refs: null,
});

const acpProfile: AgentProfileSummary = {
  id: "id-acp",
  name: "my-claude",
  agent_kind: "acp",
  revision: 0,
  llm_profile_ref: null,
  mcp_server_refs: null,
};

describe("getAgentProfileLlmDrift", () => {
  it("reports the LLM profile a launch will really use when the ref has drifted", () => {
    expect(
      getAgentProfileLlmDrift(openHandsProfile("profile-a"), true, "profile-b"),
    ).toBe("profile-b");
  });

  it.each([
    [
      "the ref and the active LLM profile agree",
      openHandsProfile("profile-a"),
      true,
      "profile-a",
    ],
    [
      "the profile is not the active one",
      openHandsProfile("profile-a"),
      false,
      "profile-b",
    ],
    [
      "the active LLM profile is unknown (still loading, or a cloud backend)",
      openHandsProfile("profile-a"),
      true,
      null,
    ],
    ["the profile is an ACP agent", acpProfile, true, "profile-b"],
    ["the profile carries no ref", openHandsProfile(null), true, "profile-b"],
  ])(
    "reports no drift when %s",
    (_case, profile, isActive, activeLlmProfile) => {
      expect(
        getAgentProfileLlmDrift(profile, isActive, activeLlmProfile),
      ).toBeNull();
    },
  );
});
