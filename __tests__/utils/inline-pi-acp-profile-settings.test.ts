import { describe, expect, it } from "vitest";
import type { ACPAgentProfile } from "@openhands/typescript-client";
import { buildInlinePiAcpAgentSettingsFromProfile } from "#/utils/inline-pi-acp-profile-settings";

describe("buildInlinePiAcpAgentSettingsFromProfile", () => {
  const baseProfile = {
    id: "profile-1",
    name: "MyPi",
    revision: 0,
    agent_kind: "acp",
    acp_server: "custom",
    acp_command: "npx -y pi-acp",
    acp_model: "default",
    acp_args: null,
    acp_session_mode: null,
    acp_prompt_timeout: 1800,
    mcp_server_refs: [],
  } satisfies ACPAgentProfile;

  it("rewrites npx pi-acp to the preinstalled binary in Docker mode", () => {
    expect(
      buildInlinePiAcpAgentSettingsFromProfile(baseProfile, "docker"),
    ).toMatchObject({
      agent_kind: "acp",
      acp_server: "custom",
      acp_command: ["pi-acp"],
      acp_model: null,
    });
  });

  it("keeps npx outside Docker", () => {
    expect(
      buildInlinePiAcpAgentSettingsFromProfile(baseProfile, "dev:automation"),
    ).toMatchObject({
      acp_command: ["npx", "-y", "pi-acp"],
      acp_model: null,
    });
  });

  it("returns null for non-Pi ACP profiles", () => {
    expect(
      buildInlinePiAcpAgentSettingsFromProfile({
        ...baseProfile,
        acp_server: "claude-code",
        acp_command: null,
      }),
    ).toBeNull();
  });
});
