import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { QueryClient } from "@tanstack/react-query";
import AgentServerConversationService, {
  type CreateConversationOptions,
} from "#/api/conversation-service/agent-server-conversation-service.api";
import { CANVAS_UI_CLIENT_TOOL } from "#/api/canvas-ui-client-tool";
import { LAUNCH_CHILD_CONVERSATION_CLIENT_TOOL } from "#/api/launch-child-conversation-client-tool";
import { AUTOMATION_FORM_UPDATE_TOOL_NAME } from "#/constants/automation-form";
import { REQUEST_SECRET_TOOL_NAME } from "#/constants/secret-request";
import { launchConversation } from "#/hooks/mutation/resolve-conversation-launch";
import {
  getFetchCall,
  getJsonBody,
  mockJsonResponse,
} from "./fetch-test-utils";

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

describe("AgentServerConversationService cloud branch", () => {
  it("createConversation POSTs the cloud payload directly and returns a WORKING task", async () => {
    fetchMock.mockResolvedValue(
      mockJsonResponse({
        id: "task-123",
        created_by_user_id: null,
        status: "WORKING",
        detail: null,
        app_conversation_id: null,
        agent_server_url: null,
        request: {},
        created_at: "2026-05-06T00:00:00Z",
        updated_at: "2026-05-06T00:00:00Z",
      }),
    );

    const result = await AgentServerConversationService.createConversation({
      initialUserMsg: "fix the bug",
      conversationInstructions: "Optional title",
      metadata: {
        selected_repository: "user/repo",
        selected_branch: "main",
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        git_provider: "github" as any,
      },
    });

    expect(fetchMock).toHaveBeenCalledOnce();
    const [url, init] = getFetchCall(fetchMock);
    expect(url).toBe(`${cloudBackend.host}/api/v1/app-conversations`);
    expect(init).toMatchObject({
      method: "POST",
      headers: { Authorization: "Bearer bearer-token" },
    });
    const requestBody = getJsonBody(init);

    // cloud payload shape — flat fields, NO encrypted-settings round-trip.
    expect(requestBody.selected_repository).toBe("user/repo");
    expect(requestBody.selected_branch).toBe("main");
    expect(requestBody.git_provider).toBe("github");
    expect(requestBody.title).toBe("Optional title");
    expect(requestBody.initial_message).toEqual({
      role: "user",
      content: [{ type: "text", text: "fix the bug" }],
    });
    // The local-only encrypted-settings keys must NOT be present.
    expect(requestBody).not.toHaveProperty("agent_settings_encrypted");
    expect(requestBody).not.toHaveProperty("conversation_settings_encrypted");

    // The returned task is the upstream task — WORKING, no app_conversation_id yet.
    expect(result.id).toBe("task-123");
    expect(result.status).toBe("WORKING");
    expect(result.app_conversation_id).toBeNull();
  });

  it("createConversation forwards the route-at-start suffix as agent_launch_additions on the cloud path", async () => {
    fetchMock.mockResolvedValue(
      mockJsonResponse({
        id: "task-124",
        created_by_user_id: null,
        status: "WORKING",
        detail: null,
        app_conversation_id: null,
        agent_server_url: null,
        request: {},
        created_at: "2026-05-06T00:00:00Z",
        updated_at: "2026-05-06T00:00:00Z",
      }),
    );

    await AgentServerConversationService.createConversation({
      initialUserMsg: "fix the bug",
      runRouterAtConversationStart: true,
      hasActiveMetaProfile: true,
    });

    const [, init] = getFetchCall(fetchMock);
    const requestBody = getJsonBody(init) as {
      agent_launch_additions?: { system_message_suffix_append?: string };
    };

    expect(requestBody.agent_launch_additions).toBeDefined();
    expect(
      requestBody.agent_launch_additions?.system_message_suffix_append,
    ).toContain("ROUTE_AT_CONVERSATION_START");
  });

  it("createConversation omits agent_launch_additions on the cloud path when the toggle is off", async () => {
    fetchMock.mockResolvedValue(
      mockJsonResponse({
        id: "task-125",
        created_by_user_id: null,
        status: "WORKING",
        detail: null,
        app_conversation_id: null,
        agent_server_url: null,
        request: {},
        created_at: "2026-05-06T00:00:00Z",
        updated_at: "2026-05-06T00:00:00Z",
      }),
    );

    await AgentServerConversationService.createConversation({
      initialUserMsg: "fix the bug",
    });

    const [, init] = getFetchCall(fetchMock);
    const requestBody = getJsonBody(init);

    expect(requestBody).not.toHaveProperty("agent_launch_additions");
  });

  it("createConversation omits agent_launch_additions on the cloud path when no meta-profile is active", async () => {
    // The toggle is on, but with no active meta-profile the agent-server
    // does not attach `route_task_to_model`, so the instruction must not be
    // emitted (the router would have nothing to call).
    fetchMock.mockResolvedValue(
      mockJsonResponse({
        id: "task-126",
        created_by_user_id: null,
        status: "WORKING",
        detail: null,
        app_conversation_id: null,
        agent_server_url: null,
        request: {},
        created_at: "2026-05-06T00:00:00Z",
        updated_at: "2026-05-06T00:00:00Z",
      }),
    );

    await AgentServerConversationService.createConversation({
      initialUserMsg: "fix the bug",
      runRouterAtConversationStart: true,
      hasActiveMetaProfile: false,
    });

    const [, init] = getFetchCall(fetchMock);
    const requestBody = getJsonBody(init);

    expect(requestBody).not.toHaveProperty("agent_launch_additions");
  });

  it("getStartTask polls /api/v1/app-conversations/start-tasks?ids= directly", async () => {
    fetchMock.mockResolvedValue(
      mockJsonResponse([
        {
          id: "task-123",
          created_by_user_id: null,
          status: "READY",
          detail: null,
          app_conversation_id: "conv-456",
          agent_server_url: "https://runtime-456.app.all-hands.dev",
          request: {},
          created_at: "2026-05-06T00:00:00Z",
          updated_at: "2026-05-06T00:00:00Z",
        },
      ]),
    );

    const result =
      await AgentServerConversationService.getStartTask("task-123");

    const [url, init] = getFetchCall(fetchMock);
    expect(url).toBe(
      `${cloudBackend.host}/api/v1/app-conversations/start-tasks?ids=task-123`,
    );
    expect(init).toMatchObject({
      method: "GET",
      headers: { Authorization: "Bearer bearer-token" },
    });
    expect(result?.status).toBe("READY");
    expect(result?.app_conversation_id).toBe("conv-456");
  });
});

describe("request_secret on a cloud launch", () => {
  const START_TASK = {
    id: "task-1",
    status: "WORKING",
    app_conversation_id: null,
    agent_server_url: null,
  };

  function arrangeCloud({
    config,
    agentProfiles = { profiles: [], active_agent_profile_id: null },
  }: {
    config: Response;
    agentProfiles?: unknown;
  }) {
    fetchMock.mockImplementation(async (url: string) => {
      const { pathname } = new URL(url);
      if (pathname === "/api/v1/web-client/config") return config;
      if (pathname === "/api/agent-profiles") {
        return mockJsonResponse(agentProfiles);
      }
      if (pathname === "/api/v1/app-conversations") {
        return mockJsonResponse(START_TASK);
      }
      throw new Error(`Unexpected request: ${url}`);
    });
  }

  async function launch(options: CreateConversationOptions = {}) {
    await launchConversation(
      {
        queryClient: new QueryClient(),
        backendId: cloudBackend.id,
        orgId: null,
        isCloud: true,
      },
      { initialUserMsg: "deploy it", ...options },
    );
    const start = fetchMock.mock.calls.find(([url]) =>
      String(url).endsWith("/api/v1/app-conversations"),
    ) as [string, RequestInit];
    return getJsonBody(start[1]) as {
      client_tools?: { name: string }[];
    };
  }

  function requestedPaths() {
    return fetchMock.mock.calls.map(([url]) => new URL(String(url)).pathname);
  }

  it("offers request_secret when the instance enables client tools", async () => {
    arrangeCloud({
      config: mockJsonResponse({
        feature_flags: { enable_client_tools: true },
      }),
    });

    const body = await launch();

    expect(body.client_tools?.map((tool) => tool.name)).toEqual([
      REQUEST_SECRET_TOOL_NAME,
    ]);
  });

  it.each([
    ["disables client tools", mockJsonResponse({ feature_flags: {} })],
    [
      "turns client tools off",
      mockJsonResponse({ feature_flags: { enable_client_tools: false } }),
    ],
    ["cannot be asked", mockJsonResponse({ detail: "boom" }, 500)],
  ])("starts without client tools when the instance %s", async (_, config) => {
    arrangeCloud({ config });

    const body = await launch();

    expect(body).not.toHaveProperty("client_tools");
  });

  it("adds request_secret to the automation setup tools", async () => {
    arrangeCloud({
      config: mockJsonResponse({
        feature_flags: { enable_client_tools: true },
      }),
    });

    const body = await launch({ automationSetup: true });

    expect(body.client_tools?.map((tool) => tool.name)).toEqual([
      CANVAS_UI_CLIENT_TOOL.name,
      LAUNCH_CHILD_CONVERSATION_CLIENT_TOOL.name,
      AUTOMATION_FORM_UPDATE_TOOL_NAME,
      REQUEST_SECRET_TOOL_NAME,
    ]);
  });

  it("gives an ACP agent no client tools", async () => {
    arrangeCloud({
      config: mockJsonResponse({
        feature_flags: { enable_client_tools: true },
      }),
      agentProfiles: {
        profiles: [
          {
            id: "acp-1",
            name: "claude",
            agent_kind: "acp",
            revision: 1,
            llm_profile_ref: null,
            mcp_server_refs: null,
          },
        ],
        active_agent_profile_id: "acp-1",
      },
    });

    const body = await launch();

    expect(body).toMatchObject({ agent_profile_id: "acp-1" });
    expect(body).not.toHaveProperty("client_tools");
    expect(requestedPaths()).not.toContain("/api/v1/web-client/config");
  });
});
