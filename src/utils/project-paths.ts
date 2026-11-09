import { projectIdKey } from "#/api/projects-service/projects-service.api";

export const PROJECTS_PATH = "/projects";

/** The Projects page with the New project dialog open. */
export const NEW_PROJECT_PATH = `${PROJECTS_PATH}?new=1`;

/** A project's page; ids are written in their plain hex form. */
export const projectPath = (projectId: string) =>
  `${PROJECTS_PATH}/${encodeURIComponent(projectIdKey(projectId))}`;
