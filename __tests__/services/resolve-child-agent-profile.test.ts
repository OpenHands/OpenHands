import { beforeEach, describe, expect, it, vi } from "vitest";

import { resolveChildAgentProfile } from "#/services/resolve-child-agent-profile";

const {
  mockListAgentProfiles,
  mockGetAgentProfile,
  mockListLlmProfiles,
  mockListCloudAgentProfiles,
  mockGetCloudAgentProfile,
  mockFetchCloudLlmProfiles,
} = vi.hoisted(() => ({
  mockListAgentProfiles: vi.fn(),
  mockGetAgentProfile: vi.fn(),
  mockListLlmProfiles: vi.fn(),
  mockListCloudAgentProfiles: vi.fn(),
  mockGetCloudAgentProfile: vi.fn(),
  mockFetchCloudLlmProfiles: vi.fn(),
}));

vi.mock("#/api/agent-profiles-service/agent-profiles-service.api", () => ({
  default: {
    listProfiles: mockListAgentProfiles,
    getProfile: mockGetAgentProfile,
  },
}));

vi.mock("#/api/profiles-service/profiles-service.api", () => ({
  default: { listProfiles: mockListLlmProfiles },
}));

vi.mock("#/api/cloud/agent-profiles-service.api", () => ({
  listCloudAgentProfiles: mockListCloudAgentProfiles,
  getCloudAgentProfile: mockGetCloudAgentProfile,
}));

vi.mock("#/api/cloud/profiles-service.api", () => ({
  fetchCloudProfilesForBackend: mockFetchCloudLlmProfiles,
}));

const cloudBackend = {
  id: "cloud-1",
  kind: "cloud" as const,
  host: "https://app.all-hands.dev",
  apiKey: "secret",
  name: "Cloud",
};

function openHandsDetail(secretRefs: string[] | null = null) {
  return {
    name: "fast",
    profile: {
      id: "profile-fast",
      name: "fast",
      agent_kind: "openhands",
      llm_profile_ref: "haiku",
      secret_refs: secretRefs,
    },
  };
}

describe("resolveChildAgentProfile", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockListAgentProfiles.mockResolvedValue({
      profiles: [
        { id: "profile-fast", name: "fast", agent_kind: "openhands" },
        { id: "profile-default", name: "default", agent_kind: "openhands" },
      ],
      active_agent_profile_id: "profile-default",
    });
    mockGetAgentProfile.mockResolvedValue(openHandsDetail());
    mockListLlmProfiles.mockResolvedValue({
      profiles: [{ name: "haiku", model: "anthropic/claude-haiku-4-5" }],
      active_profile: "haiku",
    });
  });

  it("resolves a local profile by name and reports its model", async () => {
    const resolved = await resolveChildAgentProfile({
      selector: "fast",
      target: "local",
    });

    expect(resolved).toEqual({
      ok: true,
      profile: {
        id: "profile-fast",
        name: "fast",
        agentKind: "openhands",
        model: "anthropic/claude-haiku-4-5",
        llmProfileRef: "haiku",
        secretRefsRestricted: false,
      },
    });
    expect(mockListCloudAgentProfiles).not.toHaveBeenCalled();
  });

  it("resolves a local profile by id", async () => {
    const resolved = await resolveChildAgentProfile({
      selector: "profile-fast",
      target: "local",
    });

    expect(resolved.ok).toBe(true);
    expect(mockGetAgentProfile).toHaveBeenCalledWith("fast");
  });

  it("marks a secret allow-list as restricted, including an empty list", async () => {
    mockGetAgentProfile.mockResolvedValue(openHandsDetail(["ONLY_THIS"]));

    const resolved = await resolveChildAgentProfile({
      selector: "fast",
      target: "local",
    });

    expect(resolved.ok && resolved.profile.secretRefsRestricted).toBe(true);

    mockGetAgentProfile.mockResolvedValue(openHandsDetail([]));
    const empty = await resolveChildAgentProfile({
      selector: "fast",
      target: "local",
    });
    expect(empty.ok && empty.profile.secretRefsRestricted).toBe(true);
  });

  it("rejects an unknown selector before any further launch work", async () => {
    const resolved = await resolveChildAgentProfile({
      selector: "missing",
      target: "local",
    });

    expect(resolved.ok).toBe(false);
    expect(!resolved.ok && resolved.guidance).toContain("Nothing was launched");
    expect(!resolved.ok && resolved.guidance).toContain("fast");
    expect(mockGetAgentProfile).not.toHaveBeenCalled();
    expect(mockListLlmProfiles).not.toHaveBeenCalled();
  });

  it("rejects a local profile when its LLM profile list cannot be read", async () => {
    mockListLlmProfiles.mockRejectedValue(new Error("llm offline"));

    const resolved = await resolveChildAgentProfile({
      selector: "fast",
      target: "local",
    });

    expect(resolved.ok).toBe(false);
    expect(!resolved.ok && resolved.error).toContain("llm offline");
    expect(!resolved.ok && resolved.guidance).toContain("Nothing was launched");
  });

  it("rejects a profile whose LLM profile is not saved", async () => {
    mockListLlmProfiles.mockResolvedValue({
      profiles: [],
      active_profile: null,
    });

    const resolved = await resolveChildAgentProfile({
      selector: "fast",
      target: "local",
    });

    expect(resolved.ok).toBe(false);
    expect(!resolved.ok && resolved.error).toContain("haiku");
    expect(!resolved.ok && resolved.guidance).toContain("Nothing was launched");
  });

  it("rejects when the profile list cannot be read", async () => {
    mockListAgentProfiles.mockRejectedValue(new Error("offline"));

    const resolved = await resolveChildAgentProfile({
      selector: "fast",
      target: "local",
    });

    expect(resolved.ok).toBe(false);
    expect(!resolved.ok && resolved.error).toContain("offline");
    expect(mockGetAgentProfile).not.toHaveBeenCalled();
  });

  it("uses an ACP profile's acp_model and does not look up an LLM profile", async () => {
    mockListAgentProfiles.mockResolvedValue({
      profiles: [{ id: "profile-codex", name: "codex", agent_kind: "acp" }],
      active_agent_profile_id: null,
    });
    mockGetAgentProfile.mockResolvedValue({
      name: "codex",
      profile: {
        id: "profile-codex",
        name: "codex",
        agent_kind: "acp",
        acp_model: "gpt-5.4-mini",
        secret_refs: null,
      },
    });

    const resolved = await resolveChildAgentProfile({
      selector: "codex",
      target: "local",
    });

    expect(resolved).toMatchObject({
      ok: true,
      profile: {
        id: "profile-codex",
        agentKind: "acp",
        model: "gpt-5.4-mini",
        llmProfileRef: null,
      },
    });
    expect(mockListLlmProfiles).not.toHaveBeenCalled();
  });

  it("resolves a cloud profile against the picked cloud backend", async () => {
    mockListCloudAgentProfiles.mockResolvedValue({
      profiles: [{ id: "profile-fast", name: "fast", agent_kind: "openhands" }],
      active_agent_profile_id: null,
    });
    mockGetCloudAgentProfile.mockResolvedValue(openHandsDetail(["ONLY_THIS"]));
    mockFetchCloudLlmProfiles.mockResolvedValue({
      profiles: [{ name: "haiku", model: "anthropic/claude-haiku-4-5" }],
      active_profile: null,
    });

    const resolved = await resolveChildAgentProfile({
      selector: "fast",
      target: "cloud",
      cloudBackend,
    });

    expect(mockListCloudAgentProfiles).toHaveBeenCalledWith(cloudBackend);
    expect(mockGetCloudAgentProfile).toHaveBeenCalledWith("fast", cloudBackend);
    expect(mockFetchCloudLlmProfiles).toHaveBeenCalledWith(cloudBackend);
    expect(mockListAgentProfiles).not.toHaveBeenCalled();
    expect(resolved).toMatchObject({
      ok: true,
      profile: {
        id: "profile-fast",
        model: "anthropic/claude-haiku-4-5",
        secretRefsRestricted: true,
      },
    });
  });

  it("launches a cloud profile whose LLM profile is not on the visible list", async () => {
    mockListCloudAgentProfiles.mockResolvedValue({
      profiles: [{ id: "profile-fast", name: "fast", agent_kind: "openhands" }],
      active_agent_profile_id: null,
    });
    mockGetCloudAgentProfile.mockResolvedValue(openHandsDetail());
    mockFetchCloudLlmProfiles.mockResolvedValue({
      profiles: [],
      active_profile: null,
    });

    const resolved = await resolveChildAgentProfile({
      selector: "fast",
      target: "cloud",
      cloudBackend,
    });

    expect(resolved.ok).toBe(true);
    expect(resolved).toMatchObject({
      ok: true,
      profile: {
        id: "profile-fast",
        model: null,
        llmProfileRef: "haiku",
      },
    });
    expect(resolved.ok && resolved.profile.modelNote).toContain(
      "not confirmed",
    );
    expect(resolved.ok && resolved.profile.modelNote).toContain(
      "Organization-only",
    );
  });

  it("still launches a cloud profile when the LLM profile list cannot be read", async () => {
    mockListCloudAgentProfiles.mockResolvedValue({
      profiles: [{ id: "profile-fast", name: "fast", agent_kind: "openhands" }],
      active_agent_profile_id: null,
    });
    mockGetCloudAgentProfile.mockResolvedValue(openHandsDetail());
    mockFetchCloudLlmProfiles.mockRejectedValue(new Error("llm offline"));

    const resolved = await resolveChildAgentProfile({
      selector: "fast",
      target: "cloud",
      cloudBackend,
    });

    expect(resolved.ok).toBe(true);
    expect(resolved.ok && resolved.profile.model).toBeNull();
    expect(resolved.ok && resolved.profile.modelNote).toContain("llm offline");
  });

  it("rejects an unknown cloud profile without reading LLM profiles", async () => {
    mockListCloudAgentProfiles.mockResolvedValue({
      profiles: [{ id: "profile-default", name: "default" }],
      active_agent_profile_id: "profile-default",
    });

    const resolved = await resolveChildAgentProfile({
      selector: "missing",
      target: "cloud",
      cloudBackend,
    });

    expect(resolved.ok).toBe(false);
    expect(mockGetCloudAgentProfile).not.toHaveBeenCalled();
    expect(mockFetchCloudLlmProfiles).not.toHaveBeenCalled();
  });
});
