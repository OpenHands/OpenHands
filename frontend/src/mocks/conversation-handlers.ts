import { http, delay, HttpResponse } from "msw";
import { Conversation, ResultSet } from "#/api/open-hands.types";

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

export const CONVERSATION_HANDLERS = [
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
];
