import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import SkillsService from "#/api/skills-service";
import { getFetchCall, mockJsonResponse } from "./fetch-test-utils";

vi.mock("@openhands/extensions/skills", () => ({
  SKILLS_CATALOG: [
    {
      name: "bundled-skill",
      description: "Bundled skill description",
      triggers: ["/bundled"],
      content: "Bundled skill content",
    },
  ],
}));

const cloudBackend: Backend = {
  id: "prod",
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "bearer-token",
  kind: "cloud",
};

const originalFetch = global.fetch;
const fetchMock = vi.fn();

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  setRegisteredBackends([cloudBackend]);
  setActiveSelection({ backendId: cloudBackend.id });
  fetchMock.mockReset();
  global.fetch = fetchMock as typeof fetch;
});

afterEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  fetchMock.mockReset();
  global.fetch = originalFetch;
});

describe("SkillsService.getSkills against cloud backend", () => {
  it("paginates /api/v1/skills/search directly and returns the merged list", async () => {
    fetchMock
      .mockResolvedValueOnce(
        mockJsonResponse({
          items: [
            { name: "alpha", type: "knowledge", source: "global" },
            {
              name: "beta",
              type: "task",
              source: "user",
              triggers: ["foo"],
            },
          ],
          next_page_id: "beta",
        }),
      )
      .mockResolvedValueOnce(
        mockJsonResponse({
          items: [{ name: "gamma", type: "knowledge", source: "user" }],
          next_page_id: null,
        }),
      );

    const skills = await SkillsService.getSkills();

    expect(fetchMock).toHaveBeenCalledTimes(2);

    const [firstUrl, firstInit] = getFetchCall(fetchMock, 0);
    expect(firstInit).toMatchObject({
      method: "GET",
      headers: { Authorization: "Bearer bearer-token" },
    });
    expect(firstUrl).toMatch(
      /^https:\/\/app\.all-hands\.dev\/api\/v1\/skills\/search\?/,
    );
    expect(firstUrl).not.toContain("page_id=");

    const [secondUrl] = getFetchCall(fetchMock, 1);
    expect(secondUrl).toContain("page_id=beta");

    expect(skills.map((s) => s.name)).toEqual(["alpha", "beta", "gamma"]);
    expect(skills[1]).toMatchObject({ triggers: ["foo"] });
  });

  it("adds bundled descriptions to matching sparse Cloud skills", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        items: [
          {
            name: "bundled-skill",
            type: "knowledge",
            source: "global",
            triggers: ["/bundled"],
          },
          {
            name: "custom-skill",
            type: "knowledge",
            source: "user",
            triggers: ["/custom"],
          },
        ],
        next_page_id: null,
      }),
    );

    const skills = await SkillsService.getSkills();

    expect(skills[0]).toMatchObject({
      name: "bundled-skill",
      source: "global",
      description: "Bundled skill description",
    });
    expect(skills[0]).not.toHaveProperty("content");
    expect(skills[1]).toEqual({
      name: "custom-skill",
      type: "knowledge",
      source: "user",
      triggers: ["/custom"],
    });
  });

  it("does not enrich same-named user or project skills", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        items: [
          {
            name: "bundled-skill",
            type: "knowledge",
            source: "user",
          },
          {
            name: "bundled-skill",
            type: "knowledge",
            source: "project",
          },
          {
            name: "bundled-skill",
            type: "knowledge",
            source: "public",
          },
        ],
        next_page_id: null,
      }),
    );

    const skills = await SkillsService.getSkills();

    expect(skills[0]).not.toHaveProperty("description");
    expect(skills[1]).not.toHaveProperty("description");
    expect(skills[2]).toMatchObject({
      source: "public",
      description: "Bundled skill description",
    });
  });

  it("preserves descriptions supplied by Cloud", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        items: [
          {
            name: "bundled-skill",
            type: "knowledge",
            source: "global",
            description: "Cloud description",
          },
        ],
        next_page_id: null,
      }),
    );

    await expect(SkillsService.getSkills()).resolves.toMatchObject([
      { description: "Cloud description" },
    ]);
  });

  it("normalizes conversation-loaded Cloud skills separately from the catalog", async () => {
    fetchMock.mockResolvedValueOnce(
      mockJsonResponse({
        skills: [
          {
            name: "custom-skill",
            type: "agentskills",
            content:
              "---\nname: custom-skill\ndescription: Custom description\n---\nInstructions",
            triggers: ["/custom"],
          },
        ],
      }),
    );

    const skills =
      await SkillsService.getConversationLoadedSkills("conversation/1");

    const [url, init] = getFetchCall(fetchMock, 0);
    expect(url).toBe(
      "https://app.all-hands.dev/api/v1/app-conversations/conversation%2F1/skills",
    );
    expect(init).toMatchObject({
      method: "GET",
      headers: { Authorization: "Bearer bearer-token" },
    });
    expect(skills).toEqual([
      {
        name: "custom-skill",
        source: null,
        description: "Custom description",
      },
    ]);
  });
});
