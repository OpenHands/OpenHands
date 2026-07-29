import { SkillsClient } from "@openhands/typescript-client/clients";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";

const { mockGetSkills, MOCK_PUBLIC_CATALOG } = vi.hoisted(() => ({
  mockGetSkills: vi.fn(),
  MOCK_PUBLIC_CATALOG: [
    {
      name: "mock-public-skill",
      description: "A mock public skill",
      triggers: ["mock"],
      content: "mock content",
    },
    {
      name: "another-public-skill",
      description: "Another one",
      triggers: [],
      content: "more content",
      license: "MIT",
    },
  ],
}));

vi.mock("@openhands/typescript-client/clients", () => ({
  SkillsClient: vi.fn(function SkillsClientMock() {
    return { getSkills: mockGetSkills };
  }),
}));

vi.mock("@openhands/extensions/skills", () => ({
  SKILLS_CATALOG: MOCK_PUBLIC_CATALOG,
}));

import SkillsService from "#/api/skills-service";

const localBackend: Backend = {
  id: "local",
  name: "Local",
  host: "http://127.0.0.1:8000",
  apiKey: "",
  kind: "local",
};

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  setRegisteredBackends([localBackend]);
  setActiveSelection({ backendId: localBackend.id });
  mockGetSkills.mockReset();
  vi.mocked(SkillsClient).mockClear();
});

afterEach(() => {
  vi.unstubAllEnvs();
  __resetActiveStoreForTests();
});

describe("SkillsService.getSkills against the agent-server backend", () => {
  it("requests only user/project skills from agent-server (load_public: false) and appends the bundled public catalog", async () => {
    const userSkill = {
      name: "my-custom-skill",
      type: "knowledge",
      content: "custom content",
      triggers: [],
      source: "user",
      is_agentskills_format: false,
    };
    mockGetSkills.mockResolvedValue({
      skills: [userSkill],
      sources: { sandbox: 0, sdk_base: 0, org: 0, project: 0 },
    });

    const skills = await SkillsService.getSkills();

    // Agent-server is asked only for user/project skills, not public.
    expect(mockGetSkills).toHaveBeenCalledTimes(1);
    expect(mockGetSkills.mock.calls[0]?.[0]).toMatchObject({
      load_public: false,
      load_user: true,
      load_project: true,
      load_org: false,
    });

    // Result = local skills first, then all bundled public skills.
    expect(skills[0]?.name).toBe("my-custom-skill");
    expect(skills).toHaveLength(1 + MOCK_PUBLIC_CATALOG.length);

    // Every public skill from the bundled catalog is present.
    const publicNames = skills.slice(1).map((s) => s.name);
    for (const entry of MOCK_PUBLIC_CATALOG) {
      expect(publicNames).toContain(entry.name);
    }
    expect(skills.slice(1).every((s) => s.source === "public")).toBe(true);
  });

  it("returns only bundled public skills when agent-server is unreachable", async () => {
    mockGetSkills.mockRejectedValue(new Error("ECONNREFUSED"));

    const skills = await SkillsService.getSkills();

    expect(skills).toHaveLength(MOCK_PUBLIC_CATALOG.length);
    expect(skills.every((s) => s.source === "public")).toBe(true);
  });
});
