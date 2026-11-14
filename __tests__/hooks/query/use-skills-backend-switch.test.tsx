import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  getActiveBackend,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { useSkills } from "#/hooks/query/use-skills";
import type { Backend } from "#/api/backend-registry/types";

vi.mock("#/api/skills-service", () => ({
  default: {
    getSkills: vi.fn(),
  },
}));

import SkillsService from "#/api/skills-service";

const getSkillsMock = vi.mocked(SkillsService.getSkills);

const BACKEND_A: Backend = {
  id: "backend-a",
  name: "Backend A",
  host: "http://localhost:3001",
  apiKey: "",
  kind: "local",
};

const BACKEND_B: Backend = {
  id: "backend-b",
  name: "Backend B",
  host: "http://localhost:3002",
  apiKey: "",
  kind: "local",
};

const Probe = () => {
  const { data } = useSkills();
  return <div data-testid="skills">{data?.map((s) => s.name).join(",")}</div>;
};

const renderProbe = () =>
  render(
    <QueryClientProvider client={new QueryClient()}>
      <ActiveBackendProvider>
        <Probe />
      </ActiveBackendProvider>
    </QueryClientProvider>,
  );

describe("useSkills across a backend switch (#16893)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.localStorage.clear();
    setRegisteredBackends([BACKEND_A, BACKEND_B]);
    setActiveSelection({ backendId: BACKEND_A.id });
    return () => __resetActiveStoreForTests();
  });

  it("refetches skills from the newly active backend instead of serving the stale cache", async () => {
    // Arrange: the mock serves whatever the currently active backend serves,
    // like the real SkillsService does.
    getSkillsMock.mockImplementation(async () => {
      const active = getActiveBackend().backend.id;
      return active === BACKEND_A.id
        ? [{ id: "1", name: "local-only-skill" } as never]
        : [{ id: "2", name: "cloud-skill" } as never];
    });

    renderProbe();

    await waitFor(() => {
      expect(screen.getByTestId("skills")).toHaveTextContent(
        "local-only-skill",
      );
    });
    expect(getSkillsMock).toHaveBeenCalledTimes(1);

    // Act: switch backends while backend A's cache is still fresh.
    setActiveSelection({ backendId: BACKEND_B.id });

    // Assert: a new query starts for backend B's identity.
    await waitFor(() => {
      expect(screen.getByTestId("skills")).toHaveTextContent("cloud-skill");
    });
    expect(getSkillsMock).toHaveBeenCalledTimes(2);
  });
});
