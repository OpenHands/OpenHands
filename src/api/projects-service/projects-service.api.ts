import axios from "axios";
import { NoBackendAvailableError } from "../agent-server-client-options";
import { getEffectiveLocalBackend } from "../backend-registry/active-store";
import {
  PROJECTS_API_PATH,
  SESSION_API_KEY_HEADER,
} from "./projects-constants";
import type {
  CreateProjectPayload,
  CreateWorktreePayload,
  Project,
  ProjectSummary,
  ProjectWorktree,
  UpdateProjectPayload,
} from "./projects-types";

const projectsAxios = axios.create();

projectsAxios.interceptors.request.use((config) => {
  const backend = getEffectiveLocalBackend();
  if (!backend) throw new NoBackendAvailableError();
  // eslint-disable-next-line no-param-reassign
  config.baseURL = backend.host;
  const apiKey = backend.apiKey?.trim();
  if (apiKey) {
    config.headers.set(SESSION_API_KEY_HEADER, apiKey);
  }
  return config;
});

export const ProjectsService = {
  listProjects: async (): Promise<ProjectSummary[]> => {
    const { data } =
      await projectsAxios.get<ProjectSummary[]>(PROJECTS_API_PATH);
    return data;
  },

  createProject: async (payload: CreateProjectPayload): Promise<Project> => {
    const { data } = await projectsAxios.post<Project>(
      PROJECTS_API_PATH,
      payload,
    );
    return data;
  },

  getProject: async (projectId: string): Promise<Project> => {
    const { data } = await projectsAxios.get<Project>(
      `${PROJECTS_API_PATH}/${projectId}`,
    );
    return data;
  },

  updateProject: async (
    projectId: string,
    payload: UpdateProjectPayload,
  ): Promise<Project> => {
    const { data } = await projectsAxios.patch<Project>(
      `${PROJECTS_API_PATH}/${projectId}`,
      payload,
    );
    return data;
  },

  deleteProject: async (projectId: string): Promise<void> => {
    await projectsAxios.delete(`${PROJECTS_API_PATH}/${projectId}`);
  },

  listWorktrees: async (projectId: string): Promise<ProjectWorktree[]> => {
    const { data } = await projectsAxios.get<ProjectWorktree[]>(
      `${PROJECTS_API_PATH}/${projectId}/worktrees`,
    );
    return data;
  },

  createWorktree: async (
    projectId: string,
    payload: CreateWorktreePayload,
  ): Promise<ProjectWorktree> => {
    const { data } = await projectsAxios.post<ProjectWorktree>(
      `${PROJECTS_API_PATH}/${projectId}/worktrees`,
      payload,
    );
    return data;
  },

  removeWorktree: async (
    projectId: string,
    worktreeId: string,
  ): Promise<void> => {
    await projectsAxios.delete(
      `${PROJECTS_API_PATH}/${projectId}/worktrees/${worktreeId}`,
    );
  },

  assignWorktree: async (
    projectId: string,
    worktreeId: string,
    agentSessionId: string,
  ): Promise<ProjectWorktree> => {
    const { data } = await projectsAxios.post<ProjectWorktree>(
      `${PROJECTS_API_PATH}/${projectId}/worktrees/${worktreeId}/assign`,
      { agent_session_id: agentSessionId },
    );
    return data;
  },
};

export default ProjectsService;
