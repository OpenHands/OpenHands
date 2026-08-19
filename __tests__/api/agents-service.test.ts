import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";

const { mockGetSubAgents } = vi.hoisted(() => ({
  mockGetSubAgents: vi.fn(),
}));

// Partial-mock the typed client barrel: override only SubAgentsClient and keep
// every other generated client intact for transitive imports.
vi.mock("@openhands/typescript-client/clients", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@openhands/typescript-client/clients")
    >();
  return {
    ...actual,
    SubAgentsClient: vi.fn(function SubAgentsClientMock() {
      return { getSubAgents: mockGetSubAgents };
    }),
  };
});

import AgentsService from "#/api/agents-service";
import { SubAgentsClient } from "@openhands/typescript-client/clients";

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
  mockGetSubAgents.mockReset();
  vi.mocked(SubAgentsClient).mockClear();
});

afterEach(() => {
  vi.unstubAllEnvs();
  __resetActiveStoreForTests();
});

describe("AgentsService.getAgents against the agent-server backend", () => {
  it("requests user/project/builtin agents and defaults project_dir to the working dir", async () => {
    const builtin = {
      name: "general-purpose",
      level: "builtin",
      tools: ["terminal"],
      system_prompt: "You are a general-purpose agent.",
    };
    mockGetSubAgents.mockResolvedValue({ agents: [builtin] });

    const agents = await AgentsService.getAgents();

    expect(mockGetSubAgents).toHaveBeenCalledTimes(1);
    expect(mockGetSubAgents.mock.calls[0]?.[0]).toMatchObject({
      load_user: true,
      load_project: true,
      load_builtin: true,
      project_dir: "workspace/project",
    });
    expect(agents).toHaveLength(1);
    expect(agents[0]?.name).toBe("general-purpose");
    expect(agents[0]?.level).toBe("builtin");
  });

  it("forwards an explicit projectDir to the client", async () => {
    mockGetSubAgents.mockResolvedValue({ agents: [] });

    await AgentsService.getAgents("/repo/workspace");

    expect(mockGetSubAgents.mock.calls[0]?.[0]).toMatchObject({
      project_dir: "/repo/workspace",
    });
  });

  it("returns an empty list when the agent-server is unreachable", async () => {
    mockGetSubAgents.mockRejectedValue(new Error("ECONNREFUSED"));

    const agents = await AgentsService.getAgents();

    expect(agents).toEqual([]);
  });
});
