/**
 * ProjectsService talks to the agent-server's Projects API (`/api/projects`).
 *
 * A project is a named group of conversations led by an optional coordinator
 * conversation. Membership is the conversation's ``project`` tag, which holds
 * the project's id in plain hex. The whole API answers 404 while the
 * `projects` experimental feature is off, and on servers without Projects.
 */
import {
  PROJECT_ATTENTION_ORDER,
  ProjectsClient,
  projectIdKey,
  type CreateProjectRequest,
  type NewProjectCoordinator,
  type Project,
  type ProjectArtifact,
  type ProjectAttention,
  type ProjectConversation,
  type ProjectConversationsResponse,
  type UpdateProjectRequest,
} from "@openhands/typescript-client/clients";

import { getAgentServerClientOptions } from "../agent-server-client-options";

export { PROJECT_ATTENTION_ORDER, projectIdKey };
export type {
  NewProjectCoordinator,
  Project,
  ProjectArtifact,
  ProjectAttention,
  ProjectConversation,
};
export type ProjectSummary = Pick<Project, "id" | "name">;
export type ProjectConversations = ProjectConversationsResponse;
export type CreateProjectInput = CreateProjectRequest;
export type UpdateProjectInput = UpdateProjectRequest;

const client = () => new ProjectsClient(getAgentServerClientOptions());

class ProjectsService {
  static async listProjects(): Promise<Project[]> {
    const projects = await client().listProjects();
    return projects.filter(
      (project) =>
        typeof project?.id === "string" && typeof project?.name === "string",
    );
  }

  static async getProject(projectId: string): Promise<Project> {
    return client().getProject(projectId);
  }

  static async createProject(input: CreateProjectInput): Promise<Project> {
    return client().createProject(input);
  }

  static async updateProject(
    projectId: string,
    input: UpdateProjectInput,
  ): Promise<Project> {
    return client().updateProject(projectId, input);
  }

  static async deleteProject(projectId: string): Promise<void> {
    await client().deleteProject(projectId);
  }

  static async listConversations(
    projectId: string,
  ): Promise<ProjectConversations> {
    return client().listConversations(projectId);
  }

  /** Add an existing conversation; it leaves any other project. */
  static async addConversation(
    projectId: string,
    conversationId: string,
  ): Promise<void> {
    await client().addConversation(projectId, conversationId);
  }

  static async removeConversation(
    projectId: string,
    conversationId: string,
  ): Promise<void> {
    await client().removeConversation(projectId, conversationId);
  }

  static async listArtifacts(projectId: string): Promise<ProjectArtifact[]> {
    return client().listArtifacts(projectId);
  }

  static async uploadArtifact(
    projectId: string,
    file: File,
  ): Promise<ProjectArtifact> {
    return client().uploadArtifact(projectId, file);
  }

  /** A text file's contents, for the preview. */
  static async readArtifactText(
    projectId: string,
    name: string,
  ): Promise<string> {
    return client().readArtifactText(projectId, name);
  }

  static async deleteArtifact(projectId: string, name: string): Promise<void> {
    await client().deleteArtifact(projectId, name);
  }
}

export default ProjectsService;
