import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import React from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import SkillsService from "#/api/skills-service";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { useSkills } from "#/hooks/query/use-skills";
import type { Backend } from "#/api/backend-registry/types";
import type { SkillInfo } from "#/types/settings";

vi.mock("#/api/skills-service", () => ({
  default: {
    getSkills: vi.fn(),
  },
}));

const localBackend: Backend = {
  id: "local-1",
  name: "Local 1",
  host: "http://localhost:8000",
  apiKey: "session-key",
  kind: "local",
};

const cloudBackend: Backend = {
  id: "cloud-1",
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "bearer-key",
  kind: "cloud",
};

const localSkill: SkillInfo = {
  name: "local-only-skill",
  type: "knowledge",
  source: "user",
  description: "A skill that only exists on the local agent-server",
  triggers: ["/local"],
  category: null,
  content: "",
  license: null,
  compatibility: null,
};

const cloudSkill: SkillInfo = {
  name: "cloud-only-skill",
  type: "knowledge",
  source: "public",
  description: "A skill that only exists on Cloud",
  triggers: ["/cloud"],
  category: null,
  content: "",
  license: null,
  compatibility: null,
};

function makeWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
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
  vi.mocked(SkillsService.getSkills).mockReset();
  setRegisteredBackends([localBackend, cloudBackend]);
  setActiveSelection({ backendId: localBackend.id });
});

afterEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
});

describe("useSkills — backend switch", () => {
  it("refetches when the active backend changes", async () => {
    vi.mocked(SkillsService.getSkills).mockResolvedValue([localSkill]);

    const { result } = renderHook(() => useSkills(), {
      wrapper: makeWrapper(),
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(SkillsService.getSkills).toHaveBeenCalledTimes(1);

    // Switching to another backend must be treated as a brand-new query —
    // the skills list is fetched from the active backend (Cloud skills API
    // or that backend's agent-server), so cached data from the previous
    // backend must not be reused.
    vi.mocked(SkillsService.getSkills).mockResolvedValue([cloudSkill]);
    setActiveSelection({ backendId: cloudBackend.id });

    await waitFor(() => {
      expect(SkillsService.getSkills).toHaveBeenCalledTimes(2);
    });
    await waitFor(() => {
      expect(result.current.data).toEqual([cloudSkill]);
    });
  });

  it("does not serve one backend's skills to another backend for the same projectDir", async () => {
    vi.mocked(SkillsService.getSkills).mockResolvedValue([localSkill]);

    const { result } = renderHook(() => useSkills("/workspace/project"), {
      wrapper: makeWrapper(),
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toEqual([localSkill]);

    vi.mocked(SkillsService.getSkills).mockResolvedValue([cloudSkill]);
    setActiveSelection({ backendId: cloudBackend.id });

    // The stale local-backend skills must not remain visible for the new
    // backend once the switch happens.
    await waitFor(() => {
      expect(result.current.data).toEqual([cloudSkill]);
    });
  });
});
