import { QueryClient } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import AgentProfilesService from "#/api/agent-profiles-service/agent-profiles-service.api";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import {
  ensureAgentHostTools,
  usableHostTools,
  withHostTools,
} from "#/hooks/mutation/ensure-agent-host-tools";
import ToolCatalogService from "#/api/tool-catalog-service/tool-catalog-service.api";
import { seedMockAgentProfiles } from "#/mocks/agent-profiles-handlers";

const localBackend: Backend = {
  id: "local-a",
  name: "Local A",
  host: "http://local-a.example",
  apiKey: "local-a-key",
  kind: "local",
  connectionRevision: 0,
};

const STANDARD = ["terminal", "file_editor", "task_tracker"];

describe("withHostTools", () => {
  it("pins the standard set when the profile stores no tools", () => {
    expect(withHostTools(null, ["start_child_conversation"], STANDARD)).toEqual(
      [...STANDARD, "start_child_conversation"].map((name) => ({
        name,
        params: {},
      })),
    );
  });

  it("appends to a custom selection and keeps the stored params", () => {
    const tools = [{ name: "terminal", params: { timeout: 30 } }];
    expect(
      withHostTools(tools, ["start_child_conversation"], STANDARD),
    ).toEqual([
      { name: "terminal", params: { timeout: 30 } },
      { name: "start_child_conversation", params: {} },
    ]);
  });

  it("leaves a profile that already carries the tool untouched", () => {
    const tools = [
      { name: "terminal", params: {} },
      { name: "start_child_conversation", params: {} },
    ];
    expect(withHostTools(tools, ["start_child_conversation"], STANDARD)).toBe(
      null,
    );
  });
});

describe("usableHostTools", () => {
  it("drops a host tool the backend cannot run", () => {
    expect(
      usableHostTools([{ name: "start_child_conversation", usable: false }]),
    ).toEqual([]);
  });

  it("ignores tools that are not host tools", () => {
    expect(usableHostTools([{ name: "terminal", usable: true }])).toEqual([]);
  });
});

describe("ensureAgentHostTools", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    __resetActiveStoreForTests();
    seedMockAgentProfiles();
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
    queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
  });

  afterEach(() => {
    __resetActiveStoreForTests();
    vi.restoreAllMocks();
  });

  it("attaches the server's child launcher to the profile once", async () => {
    const save = vi.spyOn(AgentProfilesService, "saveProfile");

    await expect(ensureAgentHostTools(queryClient, "default")).resolves.toBe(
      true,
    );
    await ensureAgentHostTools(queryClient, "default");

    expect(save).toHaveBeenCalledTimes(1);
    const { profile } = await AgentProfilesService.getProfile("default");
    expect(profile.agent_kind).toBe("openhands");
    expect(
      profile.agent_kind === "openhands" ? profile.tools : null,
    ).toContainEqual({ name: "start_child_conversation", params: {} });
  });

  it("keeps a custom tool selection and adds the launcher", async () => {
    await ensureAgentHostTools(queryClient, "research");

    const { profile } = await AgentProfilesService.getProfile("research");
    const tools = profile.agent_kind === "openhands" ? profile.tools : null;
    expect(tools?.map((tool) => tool.name)).toEqual([
      "task_tool_set",
      "start_child_conversation",
      "task_tracker",
    ]);
  });

  it("does not fail the launch when the catalog is unreachable", async () => {
    vi.spyOn(ToolCatalogService, "getCatalog").mockRejectedValue(
      new Error("boom"),
    );

    await expect(ensureAgentHostTools(queryClient, "default")).resolves.toBe(
      false,
    );
    const { profile } = await AgentProfilesService.getProfile("default");
    expect(
      profile.agent_kind === "openhands" ? profile.tools : null,
    ).toBeUndefined();
  });
});
