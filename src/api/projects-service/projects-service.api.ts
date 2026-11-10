/**
 * ProjectsService talks to the agent-server's Projects API (`/api/projects`).
 *
 * A project is a named group of conversations led by an optional coordinator
 * conversation. Membership is the conversation's ``project`` tag, which holds
 * the project's id in plain hex. The whole API answers 404 while the
 * `projects` experimental feature is off, and on servers without Projects.
 */
import { AgentServerClient } from "@openhands/typescript-client/clients";

import { getAgentServerClientOptions } from "../agent-server-client-options";

export interface ProjectSummary {
  id: string;
  name: string;
}

export interface Project extends ProjectSummary {
  description: string;
  coordinator_conversation_id: string | null;
  worker_agent_profile_id: string | null;
  created_at: string;
  updated_at: string;
}

/** What a conversation needs from the person, most urgent first. */
export const PROJECT_ATTENTION_ORDER = [
  "needs_you",
  "needs_input",
  "working",
  "finished",
  "idle",
  "resolved",
] as const;

export type ProjectAttention = (typeof PROJECT_ATTENTION_ORDER)[number];

export interface ProjectConversation {
  conversation_id: string;
  title: string | null;
  execution_status: string;
  attention: ProjectAttention;
  resolved: "user" | "auto" | null;
  /** Why the server resolved it (`0.93 nothing`, `0.95 critic`); newer servers. */
  resolve_note?: string | null;
  turn: "asked" | "answered" | null;
  ask: string | null;
  automation_id: string | null;
  parent_conversation_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface ProjectConversations {
  project_id: string;
  coordinator: ProjectConversation | null;
  conversations: ProjectConversation[];
}

export interface ProjectArtifact {
  name: string;
  size: number;
  /** `user` for an upload, `agent` for a file a conversation saved. */
  source: "user" | "agent";
  conversation_id: string | null;
  created_at: string;
  updated_at: string;
}

/** A coordinator conversation the server starts together with the project. */
export interface NewProjectCoordinator {
  working_dir: string;
  agent_profile_id?: string;
  title?: string;
}

export interface CreateProjectInput {
  name: string;
  description?: string;
  /** Start a new coordinator; mutually exclusive with an existing one. */
  coordinator?: NewProjectCoordinator;
  coordinator_conversation_id?: string | null;
  worker_agent_profile_id?: string | null;
}

export type UpdateProjectInput = Partial<
  Omit<CreateProjectInput, "coordinator">
>;

const client = () => new AgentServerClient(getAgentServerClientOptions());

const projectPath = (projectId: string, rest = "") =>
  `/api/projects/${encodeURIComponent(projectId)}${rest}`;

const artifactPath = (projectId: string, name?: string) =>
  projectPath(
    projectId,
    `/artifacts${name === undefined ? "" : `/${encodeURIComponent(name)}`}`,
  );

/** Ids compare without dashes or case: tags store the plain hex form. */
export const projectIdKey = (id: string) => id.replace(/-/g, "").toLowerCase();

class ProjectsService {
  static async listProjects(): Promise<Project[]> {
    const response = await client().request<{ projects?: Project[] }>({
      method: "GET",
      path: "/api/projects",
    });
    return (response?.projects ?? []).filter(
      (project) =>
        typeof project?.id === "string" && typeof project?.name === "string",
    );
  }

  static async getProject(projectId: string): Promise<Project> {
    return client().request<Project>({
      method: "GET",
      path: projectPath(projectId),
    });
  }

  static async createProject(input: CreateProjectInput): Promise<Project> {
    return client().request<Project>({
      method: "POST",
      path: "/api/projects",
      body: input,
    });
  }

  static async updateProject(
    projectId: string,
    input: UpdateProjectInput,
  ): Promise<Project> {
    return client().request<Project>({
      method: "PATCH",
      path: projectPath(projectId),
      body: input,
    });
  }

  static async deleteProject(projectId: string): Promise<void> {
    await client().request({ method: "DELETE", path: projectPath(projectId) });
  }

  static async listConversations(
    projectId: string,
  ): Promise<ProjectConversations> {
    const response = await client().request<ProjectConversations>({
      method: "GET",
      path: projectPath(projectId, "/conversations"),
    });
    return {
      project_id: response?.project_id ?? projectId,
      coordinator: response?.coordinator ?? null,
      conversations: response?.conversations ?? [],
    };
  }

  /** Add an existing conversation; it leaves any other project. */
  static async addConversation(
    projectId: string,
    conversationId: string,
  ): Promise<void> {
    await client().request({
      method: "PUT",
      path: projectPath(
        projectId,
        `/conversations/${encodeURIComponent(conversationId)}`,
      ),
    });
  }

  static async removeConversation(
    projectId: string,
    conversationId: string,
  ): Promise<void> {
    await client().request({
      method: "DELETE",
      path: projectPath(
        projectId,
        `/conversations/${encodeURIComponent(conversationId)}`,
      ),
    });
  }

  static async listArtifacts(projectId: string): Promise<ProjectArtifact[]> {
    const response = await client().request<{ artifacts?: ProjectArtifact[] }>({
      method: "GET",
      path: artifactPath(projectId),
    });
    return response?.artifacts ?? [];
  }

  static async uploadArtifact(
    projectId: string,
    file: File,
  ): Promise<ProjectArtifact> {
    const body = new FormData();
    body.append("file", file, file.name);
    return client().request<ProjectArtifact>({
      method: "POST",
      path: artifactPath(projectId),
      body,
    });
  }

  /** A text file's contents, for the preview. */
  static async readArtifactText(
    projectId: string,
    name: string,
  ): Promise<string> {
    const response = await client().request<unknown>({
      method: "GET",
      path: artifactPath(projectId, name),
      responseType: "text",
    });
    return typeof response === "string"
      ? response
      : JSON.stringify(response, null, 2);
  }

  static async deleteArtifact(projectId: string, name: string): Promise<void> {
    await client().request({
      method: "DELETE",
      path: artifactPath(projectId, name),
    });
  }
}

export default ProjectsService;
