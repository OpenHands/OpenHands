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
  V1AppConversationStartTaskPage,
  V1AppConversationPage,
} from "#/api/conversation-service/v1-conversation-service.types";

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
const V1_CONVERSATIONS = new Map<string, V1AppConversation>();
const START_TASKS = new Map<string, V1AppConversationStartTask>();

let nextConversationNumber = 4;
let nextTaskNumber = 1;

const sortByLastUpdated = (items: Conversation[]) =>
  [...items].sort((left, right) =>
    right.last_updated_at.localeCompare(left.last_updated_at),
  );

const createMockConversation = (
  overrides: Partial<Conversation> = {},
): Conversation => {
  const now = new Date().toISOString();
  const conversationId =
    overrides.conversation_id ?? `mock-conversation-${nextConversationNumber++}`;

  return {
    conversation_id: conversationId,
    title: overrides.title ?? "New Conversation",
    selected_repository: overrides.selected_repository ?? null,
    git_provider: overrides.git_provider ?? null,
    selected_branch: overrides.selected_branch ?? null,
    last_updated_at: overrides.last_updated_at ?? now,
    created_at: overrides.created_at ?? now,
    status: overrides.status ?? "RUNNING",
    runtime_status: overrides.runtime_status ?? "STATUS$READY",
    url: overrides.url ?? null,
    session_api_key: overrides.session_api_key ?? null,
    trigger: overrides.trigger,
    pr_number: overrides.pr_number ?? null,
    public: overrides.public,
    conversation_version: overrides.conversation_version,
    sandbox_id: overrides.sandbox_id ?? null,
    sub_conversation_ids: overrides.sub_conversation_ids,
  };
};

const upsertConversation = (conversation: Conversation) => {
  CONVERSATIONS.set(conversation.conversation_id, conversation);
  return conversation;
};

const ensureConversation = (
  conversationId: string,
  overrides: Partial<Conversation> = {},
) => {
  const existingConversation = CONVERSATIONS.get(conversationId);
  if (existingConversation) {
    return existingConversation;
  }

  const recoveredConversation = createMockConversation({
    conversation_id: conversationId,
    title: "Recovered Mock Conversation",
    ...overrides,
  });

  return upsertConversation(recoveredConversation);
};

const updateConversation = (
  conversationId: string,
  updates: Partial<Conversation>,
) => {
  const currentConversation = ensureConversation(conversationId);
  return upsertConversation({
    ...currentConversation,
    ...updates,
    conversation_id: conversationId,
    last_updated_at: new Date().toISOString(),
  });
};

const toV1Conversation = (
  conversation: Conversation,
  overrides: Partial<V1AppConversation> = {},
): V1AppConversation => ({
  id: conversation.conversation_id,
  created_by_user_id: null,
  sandbox_id:
    overrides.sandbox_id ??
    conversation.sandbox_id ??
    `sandbox-${conversation.conversation_id}`,
  selected_repository: conversation.selected_repository,
  selected_branch: conversation.selected_branch,
  git_provider: conversation.git_provider,
  title: conversation.title,
  trigger: conversation.trigger ?? null,
  pr_number: conversation.pr_number ?? [],
  llm_model: null,
  metrics: null,
  created_at: conversation.created_at,
  updated_at: conversation.last_updated_at,
  sandbox_status:
    conversation.status === "STOPPED"
      ? "STOPPED"
      : conversation.status === "STARTING"
        ? "STARTING"
        : "RUNNING",
  execution_status: conversation.status === "STOPPED" ? "STOPPED" : "RUNNING",
  conversation_url: conversation.url,
  session_api_key: conversation.session_api_key,
  public: conversation.public ?? false,
  ...overrides,
});

const upsertV1Conversation = (conversation: V1AppConversation) => {
  V1_CONVERSATIONS.set(conversation.id, conversation);

  updateConversation(conversation.id, {
    title: conversation.title ?? "New Conversation",
    selected_repository: conversation.selected_repository,
    selected_branch: conversation.selected_branch,
    git_provider: conversation.git_provider,
    url: conversation.conversation_url,
    session_api_key: conversation.session_api_key,
    conversation_version: "V1",
    sandbox_id: conversation.sandbox_id,
    public: conversation.public,
    status: conversation.execution_status === "STOPPED" ? "STOPPED" : "RUNNING",
    runtime_status:
      conversation.execution_status === "STOPPED" ? null : "STATUS$READY",
  });

  return conversation;
};

const ensureV1Conversation = (
  conversationId: string,
  overrides: Partial<V1AppConversation> = {},
) => {
  const existingConversation = V1_CONVERSATIONS.get(conversationId);
  if (existingConversation) {
    return existingConversation;
  }

  const baseConversation = ensureConversation(conversationId, {
    conversation_version: "V1",
    sandbox_id: overrides.sandbox_id ?? `sandbox-${conversationId}`,
  });

  return upsertV1Conversation(toV1Conversation(baseConversation, overrides));
};

const createStartTask = (
  request: V1AppConversationStartRequest,
  appConversationId: string,
  sandboxId: string,
): V1AppConversationStartTask => {
  const now = new Date().toISOString();

  const task: V1AppConversationStartTask = {
    id: `mock-start-task-${nextTaskNumber++}`,
    created_by_user_id: null,
    status: "READY",
    detail: null,
    app_conversation_id: appConversationId,
    sandbox_id: sandboxId,
    agent_server_url: null,
    request,
    created_at: now,
    updated_at: now,
  };

  START_TASKS.set(task.id, task);
  return task;
};

export const CONVERSATION_HANDLERS = [
  http.get("/api/conversations", async () => {
    const values = sortByLastUpdated(Array.from(CONVERSATIONS.values()));
    const results: ResultSet<Conversation> = {
      results: values,
      next_page_id: null,
    };
    return HttpResponse.json(results);
  }),

  http.get("/api/conversations/:conversationId", async ({ params }) => {
    const conversationId = params.conversationId as string;
    const conversation = ensureConversation(conversationId);
    return HttpResponse.json(conversation);
  }),

  http.post("/api/conversations", async ({ request }) => {
    await delay();
    const body = (await request.json().catch(() => null)) as {
      repository?: string;
      git_provider?: Conversation["git_provider"];
      selected_branch?: string;
    } | null;

    const conversation = createMockConversation({
      title: "New Conversation",
      selected_repository: body?.repository ?? null,
      git_provider: body?.git_provider ?? null,
      selected_branch: body?.selected_branch ?? null,
      sandbox_id: `sandbox-mock-${nextConversationNumber}`,
    });

    upsertConversation(conversation);
    return HttpResponse.json(conversation, { status: 201 });
  }),

  http.patch(
    "/api/conversations/:conversationId",
    async ({ params, request }) => {
      const conversationId = params.conversationId as string;
      const conversation = ensureConversation(conversationId);
      const body = await request.json();

      if (conversation && typeof body === "object" && body?.title) {
        updateConversation(conversationId, {
          title: body.title,
        });
        return HttpResponse.json(null, { status: 200 });
      }

      return HttpResponse.json(null, { status: 404 });
    },
  ),

  http.delete("/api/conversations/:conversationId", async ({ params }) => {
    const conversationId = params.conversationId as string;
    CONVERSATIONS.delete(conversationId);
    V1_CONVERSATIONS.delete(conversationId);
    return HttpResponse.json(null, { status: 200 });
  }),

  http.get("/api/conversations/:conversationId/config", async ({ params }) => {
    const conversationId = params.conversationId as string;
    const conversation = ensureConversation(conversationId);
    const runtimeId = conversation.sandbox_id ?? `runtime-${conversationId}`;

    updateConversation(conversationId, { sandbox_id: runtimeId });
    return HttpResponse.json({ runtime_id: runtimeId });
  }),

  http.get(
    "/api/conversations/:conversationId/web-hosts",
    async () => HttpResponse.json({ hosts: {} }),
  ),

  http.get("/api/conversations/:conversationId/trajectory", async () =>
    HttpResponse.json({ trajectory: [] }),
  ),

  http.post("/api/conversations/:conversationId/start", async ({ params }) => {
    const conversationId = params.conversationId as string;
    const conversation = updateConversation(conversationId, {
      status: "RUNNING",
      runtime_status: "STATUS$READY",
    });
    return HttpResponse.json(conversation);
  }),

  http.post("/api/conversations/:conversationId/stop", async ({ params }) => {
    const conversationId = params.conversationId as string;
    const conversation = updateConversation(conversationId, {
      status: "STOPPED",
      runtime_status: null,
    });
    return HttpResponse.json(conversation);
  }),

  http.post(
    "/api/v1/app-conversations",
    async ({ request }) => {
      await delay();
      const body = (await request.json()) as V1AppConversationStartRequest;

      const conversation = createMockConversation({
        conversation_id: `mock-conversation-${nextConversationNumber++}`,
        title: body.title ?? "New Conversation",
        selected_repository: body.selected_repository ?? null,
        selected_branch: body.selected_branch ?? null,
        git_provider: body.git_provider ?? null,
        conversation_version: "V1",
        sandbox_id: body.sandbox_id ?? `sandbox-mock-${nextConversationNumber}`,
      });

      upsertConversation(conversation);
      const v1Conversation = upsertV1Conversation(
        toV1Conversation(conversation, {
          llm_model: body.llm_model ?? null,
        }),
      );
      const task = createStartTask(body, v1Conversation.id, v1Conversation.sandbox_id);

      return HttpResponse.json(task, { status: 201 });
    },
  ),

  http.get("/api/v1/app-conversations/start-tasks", async ({ request }) => {
    const url = new URL(request.url);
    const ids = url.searchParams.getAll("ids");
    const tasks = ids.map((id) => START_TASKS.get(id) ?? null);
    return HttpResponse.json(tasks);
  }),

  http.get(
    "/api/v1/app-conversations/start-tasks/search",
    async () => {
      const page: V1AppConversationStartTaskPage = {
        items: Array.from(START_TASKS.values()).sort((left, right) =>
          right.updated_at.localeCompare(left.updated_at),
        ),
        next_page_id: null,
      };
      return HttpResponse.json(page);
    },
  ),

  http.get("/api/v1/app-conversations", async ({ request }) => {
    const url = new URL(request.url);
    const ids = url.searchParams.getAll("ids");
    const conversationsPage = ids.map((id) => ensureV1Conversation(id));
    return HttpResponse.json(conversationsPage);
  }),

  http.get("/api/v1/app-conversations/search", async ({ request }) => {
    const url = new URL(request.url);
    const sandboxId = url.searchParams.get("sandbox_id__eq");
    const items = Array.from(V1_CONVERSATIONS.values()).filter(
      (conversation) => !sandboxId || conversation.sandbox_id === sandboxId,
    );

    const page: V1AppConversationPage = {
      items,
      next_page_id: null,
    };

    return HttpResponse.json(page);
  }),

  http.patch(
    "/api/v1/app-conversations/:conversationId",
    async ({ params, request }) => {
      const conversationId = params.conversationId as string;
      const currentConversation = ensureV1Conversation(conversationId);
      const body = (await request.json()) as Partial<V1AppConversation>;

      const updatedConversation = upsertV1Conversation({
        ...currentConversation,
        ...body,
        id: conversationId,
        updated_at: new Date().toISOString(),
      });

      return HttpResponse.json(updatedConversation);
    },
  ),

  http.get(
    "/api/v1/app-conversations/:conversationId/skills",
    async () => HttpResponse.json({ skills: [] }),
  ),

  http.get(
    "/api/v1/app-conversations/:conversationId/hooks",
    async () => HttpResponse.json({ hooks: [] }),
  ),

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
