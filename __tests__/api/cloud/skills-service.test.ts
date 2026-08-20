import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import SkillsService from "#/api/skills-service";
import { getFetchCall, mockJsonResponse } from "./fetch-test-utils";

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
});
