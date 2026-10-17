import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import SkillsService from "#/api/skills-service";
import {
  __resetActiveStoreForTests,
  getActiveBackend,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { useSkills } from "#/hooks/query/use-skills";
import { SKILLS_QUERY_KEYS } from "#/hooks/query/query-keys";
import type { SkillInfo } from "#/types/settings";

const firstBackend: Backend = {
  id: "local-a",
  name: "Local A",
  host: "http://localhost:8000",
  apiKey: "session-a",
  kind: "local",
};

const secondBackend: Backend = {
  id: "local-b",
  name: "Local B",
  host: "http://localhost:8001",
  apiKey: "session-b",
  kind: "local",
};

function makeSkill(name: string): SkillInfo {
  return {
    name,
    type: "agentskills",
    source: null,
    content: `# ${name}`,
    triggers: [],
  };
}

function makeWrapper(queryClient: QueryClient) {
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <ActiveBackendProvider>{children}</ActiveBackendProvider>
      </QueryClientProvider>
    );
  };
}

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  setRegisteredBackends([firstBackend, secondBackend]);
});

afterEach(() => {
  vi.restoreAllMocks();
  window.localStorage.clear();
  __resetActiveStoreForTests();
});

describe("useSkills", () => {
  it("refetches fresh skills after switching backends within the stale window", async () => {
    const getSkills = vi
      .spyOn(SkillsService, "getSkills")
      .mockImplementation(async () => {
        return [makeSkill(getActiveBackend().backend.id)];
      });
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });

    const { result } = renderHook(() => useSkills("/workspace/project"), {
      wrapper: makeWrapper(queryClient),
    });

    await waitFor(() =>
      expect(result.current.data).toEqual([makeSkill(firstBackend.id)]),
    );

    setActiveSelection({ backendId: secondBackend.id });

    await waitFor(() =>
      expect(result.current.data).toEqual([makeSkill(secondBackend.id)]),
    );
    expect(getSkills).toHaveBeenCalledTimes(2);
    expect(
      queryClient.getQueryData(
        SKILLS_QUERY_KEYS.catalog(firstBackend.id, null, "/workspace/project"),
      ),
    ).toEqual([makeSkill(firstBackend.id)]);
    expect(
      queryClient.getQueryData(
        SKILLS_QUERY_KEYS.catalog(secondBackend.id, null, "/workspace/project"),
      ),
    ).toEqual([makeSkill(secondBackend.id)]);
  });
});
