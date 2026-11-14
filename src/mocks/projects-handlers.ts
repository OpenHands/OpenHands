import { http, HttpResponse } from "msw";
import type {
  Project,
  ProjectArtifact,
  ProjectConversation,
  ProjectMemoryFile,
} from "#/api/projects-service/projects-service.api";

/**
 * MSW handlers for the agent-server's Projects API (`/api/projects`) and the
 * experimental feature switches that turn it on, so mock mode can show the
 * Projects list, a board in every state, files and memory.
 *
 * State lives in memory and resets on every page load.
 */

const NOW = "2026-11-12T10:00:00Z";
const DEMO_PROJECT_ID = "aa000000-0000-4000-8000-000000000001";
const COORDINATOR_ID = "cc000000-0000-4000-8000-000000000001";

const key = (id: string) => id.replace(/-/g, "").toLowerCase();

const conversation = (
  id: string,
  title: string,
  extra: Partial<ProjectConversation>,
): ProjectConversation => ({
  conversation_id: id,
  title,
  execution_status: "finished",
  attention: "idle",
  resolved: null,
  resolve_note: null,
  turn: "answered",
  ask: null,
  automation_id: null,
  parent_conversation_id: COORDINATOR_ID,
  created_at: NOW,
  updated_at: NOW,
  ...extra,
});

interface MockProject {
  project: Project;
  coordinator: ProjectConversation | null;
  conversations: ProjectConversation[];
  artifacts: Array<ProjectArtifact & { text: string }>;
  memory: Array<ProjectMemoryFile & { content: string }>;
}

const seed = (): MockProject[] => [
  {
    project: {
      id: DEMO_PROJECT_ID,
      name: "Release 0.2",
      description: "Get the next release ready for people to try.",
      coordinator_conversation_id: COORDINATOR_ID,
      worker_agent_profile_id: null,
      created_at: NOW,
      updated_at: NOW,
    },
    coordinator: conversation(COORDINATOR_ID, "Release 0.2", {
      parent_conversation_id: null,
    }),
    conversations: [
      conversation("01000000-0000-4000-8000-000000000001", "Changelog", {
        attention: "needs_input",
        turn: "asked",
        ask: "Should the changelog cover the beta, or only the final release?",
      }),
      conversation("02000000-0000-4000-8000-000000000001", "Publish script", {
        execution_status: "waiting_for_confirmation",
        attention: "needs_you",
        turn: null,
      }),
      conversation("03000000-0000-4000-8000-000000000001", "README", {
        execution_status: "running",
        attention: "working",
        turn: null,
      }),
      conversation("04000000-0000-4000-8000-000000000001", "Install docs", {}),
      conversation("05000000-0000-4000-8000-000000000001", "Locales", {
        attention: "resolved",
        resolved: "auto",
        resolve_note: "0.93 nothing",
      }),
    ],
    artifacts: [
      {
        name: "release-notes.md",
        size: 42,
        source: "agent",
        conversation_id: "01000000-0000-4000-8000-000000000001",
        created_at: NOW,
        updated_at: NOW,
        text: "# Release 0.2\n\n- Projects\n- Reactions\n",
      },
    ],
    memory: [
      {
        name: "MEMORY.md",
        size: 61,
        revision: "r1",
        updated_at: NOW,
        content: "- Releases are cut from main.\n- [Testing](testing.md)\n",
      },
      {
        name: "testing.md",
        size: 38,
        revision: "r1",
        updated_at: NOW,
        content: "Run the full suite before tagging.\n",
      },
    ],
  },
];

let projects: MockProject[] = seed();
// On in mock mode, so the pages show; off under test like a real server's
// default, so suites that do not ask for Projects never see it.
const defaultFeatures = (): Record<string, boolean> => ({
  projects: import.meta.env.MODE !== "test",
  conversation_reactions: false,
});

let features = defaultFeatures();

const FEATURE_TEXT: Record<string, { label: string; description: string }> = {
  conversation_reactions: {
    label: "Conversation reactions",
    description: "The agent and you can react to messages with an emoji.",
  },
  projects: {
    label: "Projects",
    description:
      "Group conversations into projects led by a coordinator conversation.",
  },
};

const feature = (featureKey: string) => ({
  key: featureKey,
  ...FEATURE_TEXT[featureKey],
  enabled: features[featureKey],
});

export function resetMockProjects(): void {
  projects = seed();
  features = defaultFeatures();
}

const notFound = (detail: string) =>
  HttpResponse.json({ detail }, { status: 404 });

/** The project a request names, or the 404 the server would answer. */
function find(projectId: string | readonly string[] | undefined) {
  if (!features.projects) return notFound("Projects are turned off");
  const found = projects.find(
    (p) => key(p.project.id) === key(String(projectId)),
  );
  return found ?? notFound("Project not found");
}

export const PROJECTS_HANDLERS = [
  http.get("*/api/settings/experimental", () =>
    HttpResponse.json(Object.keys(features).map(feature)),
  ),

  http.patch(
    "*/api/settings/experimental/:key",
    async ({ params, request }) => {
      const featureKey = String(params.key);
      if (!(featureKey in features)) return notFound("Unknown feature");
      const body = (await request.json()) as { enabled?: boolean };
      features[featureKey] = !!body.enabled;
      return HttpResponse.json(feature(featureKey));
    },
  ),

  http.get("*/api/projects", () =>
    features.projects
      ? HttpResponse.json({ projects: projects.map((p) => p.project) })
      : notFound("Projects are turned off"),
  ),

  http.post("*/api/projects", async ({ request }) => {
    if (!features.projects) return notFound("Projects are turned off");
    const body = (await request.json()) as Partial<Project> & {
      coordinator?: { title?: string };
    };
    const id = crypto.randomUUID();
    const coordinatorId = body.coordinator ? crypto.randomUUID() : null;
    const project: Project = {
      id,
      name: body.name ?? "",
      description: body.description ?? "",
      coordinator_conversation_id: coordinatorId,
      worker_agent_profile_id: body.worker_agent_profile_id ?? null,
      created_at: NOW,
      updated_at: NOW,
    };
    projects.push({
      project,
      coordinator: coordinatorId
        ? conversation(coordinatorId, body.coordinator?.title ?? project.name, {
            execution_status: "idle",
            turn: null,
            parent_conversation_id: null,
          })
        : null,
      conversations: [],
      artifacts: [],
      memory: [],
    });
    return HttpResponse.json(project, { status: 201 });
  }),

  http.get("*/api/projects/:projectId", ({ params }) => {
    const found = find(params.projectId);
    return found instanceof Response ? found : HttpResponse.json(found.project);
  }),

  http.patch("*/api/projects/:projectId", async ({ params, request }) => {
    const found = find(params.projectId);
    if (found instanceof Response) return found;
    const body = (await request.json()) as Partial<Project>;
    found.project = { ...found.project, ...body, id: found.project.id };
    return HttpResponse.json(found.project);
  }),

  http.delete("*/api/projects/:projectId", ({ params }) => {
    const found = find(params.projectId);
    if (found instanceof Response) return found;
    projects = projects.filter((p) => p !== found);
    return HttpResponse.json({ success: true });
  }),

  http.get("*/api/projects/:projectId/conversations", ({ params }) => {
    const found = find(params.projectId);
    if (found instanceof Response) return found;
    return HttpResponse.json({
      project_id: found.project.id,
      coordinator: found.coordinator,
      conversations: found.conversations,
    });
  }),

  http.put(
    "*/api/projects/:projectId/conversations/:conversationId",
    ({ params }) => {
      const found = find(params.projectId);
      if (found instanceof Response) return found;
      const id = String(params.conversationId);
      if (
        !found.conversations.some((c) => key(c.conversation_id) === key(id))
      ) {
        found.conversations.push(
          conversation(id, null as unknown as string, {
            execution_status: "idle",
            turn: null,
            parent_conversation_id: null,
          }),
        );
      }
      return HttpResponse.json(found.project);
    },
  ),

  http.delete(
    "*/api/projects/:projectId/conversations/:conversationId",
    ({ params }) => {
      const found = find(params.projectId);
      if (found instanceof Response) return found;
      const id = key(String(params.conversationId));
      found.conversations = found.conversations.filter(
        (c) => key(c.conversation_id) !== id,
      );
      return HttpResponse.json({ success: true });
    },
  ),

  http.post(
    "*/api/projects/:projectId/conversations/:conversationId/:action",
    ({ params }) => {
      const found = find(params.projectId);
      if (found instanceof Response) return found;
      const id = key(String(params.conversationId));
      const member = found.conversations.find(
        (c) => key(c.conversation_id) === id,
      );
      if (!member || !["resolve", "reopen"].includes(String(params.action))) {
        return notFound("Conversation not in project");
      }
      const resolved = params.action === "resolve";
      member.resolved = resolved ? "user" : null;
      member.resolve_note = null;
      member.attention = resolved ? "resolved" : "idle";
      return new HttpResponse(null, { status: 204 });
    },
  ),

  http.get("*/api/projects/:projectId/artifacts", ({ params }) => {
    const found = find(params.projectId);
    if (found instanceof Response) return found;
    return HttpResponse.json({
      artifacts: found.artifacts.map(
        ({ text: _text, ...artifact }) => artifact,
      ),
    });
  }),

  http.post(
    "*/api/projects/:projectId/artifacts",
    async ({ params, request }) => {
      const found = find(params.projectId);
      if (found instanceof Response) return found;
      const file = (await request.formData()).get("file");
      if (!(file instanceof File)) {
        return HttpResponse.json({ detail: "No file" }, { status: 400 });
      }
      const artifact = {
        name: file.name,
        size: file.size,
        source: "user" as const,
        conversation_id: null,
        created_at: NOW,
        updated_at: new Date().toISOString(),
        text: await file.text(),
      };
      found.artifacts = [
        ...found.artifacts.filter((a) => a.name !== artifact.name),
        artifact,
      ];
      const { text: _text, ...body } = artifact;
      return HttpResponse.json(body, { status: 201 });
    },
  ),

  http.get("*/api/projects/:projectId/artifacts/:name", ({ params }) => {
    const found = find(params.projectId);
    if (found instanceof Response) return found;
    const artifact = found.artifacts.find((a) => a.name === params.name);
    return artifact
      ? HttpResponse.text(artifact.text)
      : notFound("Artifact not found");
  }),

  http.delete("*/api/projects/:projectId/artifacts/:name", ({ params }) => {
    const found = find(params.projectId);
    if (found instanceof Response) return found;
    found.artifacts = found.artifacts.filter((a) => a.name !== params.name);
    return HttpResponse.json({ success: true });
  }),

  http.get("*/api/projects/:projectId/memory", ({ params }) => {
    const found = find(params.projectId);
    if (found instanceof Response) return found;
    return HttpResponse.json({
      files: found.memory.map(({ content: _content, ...file }) => file),
    });
  }),

  http.get("*/api/projects/:projectId/memory/:name", ({ params }) => {
    const found = find(params.projectId);
    if (found instanceof Response) return found;
    const file = found.memory.find((m) => m.name === params.name);
    return file ? HttpResponse.json(file) : notFound("Memory file not found");
  }),
];
