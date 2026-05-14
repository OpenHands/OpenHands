import { http, delay, HttpResponse } from "msw";
import {
  Conversation,
  GetMicroagentsResponse,
  ResultSet,
} from "#/api/open-hands.types";
import {
  V1AppConversation,
  V1AppConversationStartRequest,
  V1AppConversationStartTask,
} from "#/api/conversation-service/v1-conversation-service.types";
import { V1ExecutionStatus } from "#/types/v1/core";

const conversations: Conversation[] = [
  {
    conversation_id: "1",
    title: "My New Project",
    selected_repository: null,
    git_provider: null,
    selected_branch: null,
    last_updated_at: new Date().toISOString(),
    created_at: new Date().toISOString(),
    status: "RUNNING",
    runtime_status: "STATUS$READY",
    url: null,
    session_api_key: null,
  },
  {
    conversation_id: "2",
    title: "Repo Testing",
    selected_repository: "octocat/hello-world",
    git_provider: "github",
    selected_branch: null,
    last_updated_at: new Date(
      Date.now() - 2 * 24 * 60 * 60 * 1000,
    ).toISOString(),
    created_at: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
    status: "STOPPED",
    runtime_status: null,
    url: null,
    session_api_key: null,
  },
  {
    conversation_id: "3",
    title: "Another Project",
    selected_repository: "octocat/earth",
    git_provider: null,
    selected_branch: "main",
    last_updated_at: new Date(
      Date.now() - 5 * 24 * 60 * 60 * 1000,
    ).toISOString(),
    created_at: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(),
    status: "STOPPED",
    runtime_status: null,
    url: null,
    session_api_key: null,
  },
];

const CONVERSATIONS = new Map<string, Conversation>(
  conversations.map((c) => [c.conversation_id, c]),
);

const v1Conversations: V1AppConversation[] = conversations.map(
  (conversation) => ({
    id: conversation.conversation_id,
    created_by_user_id: "1",
    sandbox_id: `sandbox-${conversation.conversation_id}`,
    selected_repository: conversation.selected_repository,
    selected_branch: conversation.selected_branch,
    git_provider: conversation.git_provider,
    title: conversation.title,
    trigger: null,
    pr_number: [],
    llm_model: null,
    metrics: null,
    created_at: conversation.created_at,
    updated_at: conversation.last_updated_at,
    sandbox_status: conversation.status === "RUNNING" ? "RUNNING" : "MISSING",
    execution_status:
      conversation.status === "RUNNING"
        ? V1ExecutionStatus.RUNNING
        : V1ExecutionStatus.FINISHED,
    conversation_url: conversation.url,
    session_api_key: conversation.session_api_key,
    public: false,
    sub_conversation_ids: [],
  }),
);

const V1_CONVERSATIONS = new Map<string, V1AppConversation>(
  v1Conversations.map((conversation) => [conversation.id, conversation]),
);

const V1_START_TASKS = new Map<string, V1AppConversationStartTask>();

function createV1Conversation(
  request: V1AppConversationStartRequest,
): V1AppConversation {
  const now = new Date().toISOString();
  const id = `mock-${crypto.randomUUID()}`;

  return {
    id,
    created_by_user_id: "1",
    sandbox_id: request.sandbox_id ?? `sandbox-${id}`,
    selected_repository: request.selected_repository ?? null,
    selected_branch: request.selected_branch ?? null,
    git_provider: request.git_provider ?? null,
    title: request.title ?? "New Conversation",
    trigger: request.trigger ?? null,
    pr_number: request.pr_number ?? [],
    llm_model: request.llm_model ?? null,
    metrics: null,
    created_at: now,
    updated_at: now,
    sandbox_status: "RUNNING",
    execution_status: V1ExecutionStatus.IDLE,
    conversation_url: null,
    session_api_key: null,
    public: false,
    sub_conversation_ids: [],
  };
}

function createV1StartTask(
  request: V1AppConversationStartRequest,
  conversation: V1AppConversation,
): V1AppConversationStartTask {
  const now = new Date().toISOString();

  return {
    id: `task-${crypto.randomUUID()}`,
    created_by_user_id: "1",
    status: "READY",
    detail: null,
    app_conversation_id: conversation.id,
    sandbox_id: conversation.sandbox_id,
    agent_server_url: null,
    request,
    created_at: now,
    updated_at: now,
  };
}

export const CONVERSATION_HANDLERS = [
  http.get("/api/v1/app-conversations/search", async () =>
    HttpResponse.json({
      items: Array.from(V1_CONVERSATIONS.values()),
      next_page_id: null,
    }),
  ),

  http.get("/api/v1/app-conversations", async ({ request }) => {
    const url = new URL(request.url);
    const ids = url.searchParams.getAll("ids");

    if (ids.length > 0) {
      return HttpResponse.json(
        ids.map((id) => V1_CONVERSATIONS.get(id) ?? null),
      );
    }

    return HttpResponse.json(Array.from(V1_CONVERSATIONS.values()));
  }),

  http.post("/api/v1/app-conversations", async ({ request }) => {
    const body =
      ((await request.json()) as V1AppConversationStartRequest | null) ?? {};
    const conversation = createV1Conversation(body);
    const task = createV1StartTask(body, conversation);

    V1_CONVERSATIONS.set(conversation.id, conversation);
    V1_START_TASKS.set(task.id, task);

    return HttpResponse.json(task, { status: 201 });
  }),

  http.get("/api/v1/app-conversations/start-tasks", async ({ request }) => {
    const url = new URL(request.url);
    const ids = url.searchParams.getAll("ids");

    return HttpResponse.json(ids.map((id) => V1_START_TASKS.get(id) ?? null));
  }),

  http.get("/api/v1/app-conversations/start-tasks/search", async () =>
    HttpResponse.json({
      items: Array.from(V1_START_TASKS.values()),
      next_page_id: null,
    }),
  ),

  http.get("/api/v1/app-conversations/:conversationId", async ({ params }) => {
    const conversationId = params.conversationId as string;
    const conversation = V1_CONVERSATIONS.get(conversationId);

    if (conversation) return HttpResponse.json(conversation);
    return HttpResponse.json(null, { status: 404 });
  }),

  http.get("/api/conversations", async () => {
    const values = Array.from(CONVERSATIONS.values());
    const results: ResultSet<Conversation> = {
      results: values,
      next_page_id: null,
    };
    return HttpResponse.json(results);
  }),

  http.get("/api/conversations/:conversationId", async ({ params }) => {
    const conversationId = params.conversationId as string;
    const project = CONVERSATIONS.get(conversationId);
    if (project) return HttpResponse.json(project);
    return HttpResponse.json(null, { status: 404 });
  }),

  http.post("/api/conversations", async () => {
    await delay();
    const conversation: Conversation = {
      conversation_id: (Math.random() * 100).toString(),
      title: "New Conversation",
      selected_repository: null,
      git_provider: null,
      selected_branch: null,
      last_updated_at: new Date().toISOString(),
      created_at: new Date().toISOString(),
      status: "RUNNING",
      runtime_status: "STATUS$READY",
      url: null,
      session_api_key: null,
    };
    CONVERSATIONS.set(conversation.conversation_id, conversation);
    return HttpResponse.json(conversation, { status: 201 });
  }),

  http.patch(
    "/api/conversations/:conversationId",
    async ({ params, request }) => {
      const conversationId = params.conversationId as string;
      const conversation = CONVERSATIONS.get(conversationId);

      if (conversation) {
        const body = await request.json();
        if (typeof body === "object" && body?.title) {
          CONVERSATIONS.set(conversationId, {
            ...conversation,
            title: body.title,
          });
          return HttpResponse.json(null, { status: 200 });
        }
      }
      return HttpResponse.json(null, { status: 404 });
    },
  ),

  http.delete("/api/conversations/:conversationId", async ({ params }) => {
    const conversationId = params.conversationId as string;
    if (CONVERSATIONS.has(conversationId)) {
      CONVERSATIONS.delete(conversationId);
      return HttpResponse.json(null, { status: 200 });
    }
    return HttpResponse.json(null, { status: 404 });
  }),

  http.post(
    "/api/v1/conversations/:conversationId/pending-messages",
    async () => HttpResponse.json({ id: "mock-pending-id", position: 0 }),
  ),

  http.get("/api/conversations/:conversationId/microagents", async () => {
    const response: GetMicroagentsResponse = {
      microagents: [
        {
          name: "init",
          type: "agentskills",
          content: "Initialize an AGENTS.md file for the repository",
          triggers: ["/init"],
        },
        {
          name: "releasenotes",
          type: "agentskills",
          content: "Generate a changelog from the most recent release",
          triggers: ["/releasenotes"],
        },
        {
          name: "test-runner",
          type: "agentskills",
          content: "Run the test suite and report results",
          triggers: ["/test"],
        },
        {
          name: "code-search",
          type: "knowledge",
          content: "Search the codebase semantically",
          triggers: ["/search"],
        },
        {
          name: "docker",
          type: "agentskills",
          content: "Docker usage guide for container environments",
          triggers: ["docker", "container"],
        },
        {
          name: "github",
          type: "agentskills",
          content: "GitHub API interaction guide",
          triggers: ["github", "git"],
        },
        {
          name: "work_hosts",
          type: "repo",
          content: "Available hosts for web applications",
          triggers: [],
        },
      ],
    };
    return HttpResponse.json(response);
  }),
];
