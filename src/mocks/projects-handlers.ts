import { http, HttpResponse } from "msw";
import { PROJECTS_API_PATH } from "#/api/projects-service/projects-constants";
import type {
  Project,
  ProjectSummary,
  ProjectWorktree,
} from "#/api/projects-service/projects-types";

let projects: Project[] = [];
let nextId = 1;

function id(prefix: string): string {
  nextId += 1;
  return `${prefix}-${nextId}`;
}

function now(): string {
  return new Date().toISOString();
}

export function resetProjectsMockData() {
  projects = [];
  nextId = 1;
}

function toSummary(project: Project): ProjectSummary {
  const { worktrees, ...summary } = project;
  return { ...summary, worktree_count: worktrees.length };
}

function getProject(projectId: string): Project | undefined {
  return projects.find((project) => project.id === projectId);
}

export const PROJECTS_HANDLERS = [
  http.get(`*${PROJECTS_API_PATH}`, () =>
    HttpResponse.json(projects.map(toSummary)),
  ),
  http.post(`*${PROJECTS_API_PATH}`, async ({ request }) => {
    const body = (await request.json()) as {
      name?: string;
      description?: string | null;
      repo_url?: string | null;
      local_path?: string | null;
      default_branch?: string | null;
      default_agent_profile?: string | null;
      kanban_board_id?: string | null;
      cost_cap?: number | null;
    };
    if (!body.name?.trim()) {
      return HttpResponse.json(
        { error: "Project name is required" },
        { status: 400 },
      );
    }
    const createdAt = now();
    const project: Project = {
      id: id("project"),
      name: body.name.trim(),
      description: body.description ?? null,
      repo_url: body.repo_url ?? null,
      local_path: body.local_path?.trim() || `/tmp/${body.name.trim()}`,
      default_branch: body.default_branch?.trim() || "main",
      default_agent_profile: body.default_agent_profile ?? null,
      kanban_board_id: body.kanban_board_id ?? null,
      cost_cap: body.cost_cap ?? null,
      status: "idle",
      worktree_count: 0,
      created_at: createdAt,
      updated_at: createdAt,
      worktrees: [],
    };
    projects.push(project);
    return HttpResponse.json(project, { status: 201 });
  }),
  http.get(`*${PROJECTS_API_PATH}/:projectId/worktrees`, ({ params }) => {
    const project = getProject(String(params.projectId));
    if (!project) {
      return HttpResponse.json({ error: "not found" }, { status: 404 });
    }
    return HttpResponse.json(project.worktrees);
  }),
  http.post(
    `*${PROJECTS_API_PATH}/:projectId/worktrees`,
    async ({ params, request }) => {
      const project = getProject(String(params.projectId));
      if (!project) {
        return HttpResponse.json({ error: "not found" }, { status: 404 });
      }
      const body = (await request.json()) as { branch_name?: string };
      if (!body.branch_name?.trim()) {
        return HttpResponse.json(
          { error: "branch_name is required" },
          { status: 400 },
        );
      }
      const createdAt = now();
      const worktree: ProjectWorktree = {
        id: id("wt"),
        project_id: project.id,
        branch_name: body.branch_name.trim(),
        path: `${project.local_path}/.worktrees/${body.branch_name.trim()}`,
        status: "idle",
        agent_session_id: null,
        created_at: createdAt,
        updated_at: createdAt,
      };
      project.worktrees.push(worktree);
      project.worktree_count = project.worktrees.length;
      project.status = "active";
      project.updated_at = createdAt;
      return HttpResponse.json(worktree, { status: 201 });
    },
  ),
  http.post(
    `*${PROJECTS_API_PATH}/:projectId/worktrees/:worktreeId/assign`,
    async ({ params, request }) => {
      const project = getProject(String(params.projectId));
      const worktree = project?.worktrees.find(
        (item) => item.id === String(params.worktreeId),
      );
      if (!project || !worktree) {
        return HttpResponse.json({ error: "not found" }, { status: 404 });
      }
      const body = (await request.json()) as { agent_session_id?: string };
      worktree.agent_session_id = body.agent_session_id?.trim() || null;
      worktree.status = "working";
      worktree.updated_at = now();
      return HttpResponse.json(worktree);
    },
  ),
  http.delete(
    `*${PROJECTS_API_PATH}/:projectId/worktrees/:worktreeId`,
    ({ params }) => {
      const project = getProject(String(params.projectId));
      if (!project) {
        return HttpResponse.json({ error: "not found" }, { status: 404 });
      }
      project.worktrees = project.worktrees.filter(
        (item) => item.id !== String(params.worktreeId),
      );
      project.worktree_count = project.worktrees.length;
      return new HttpResponse(null, { status: 204 });
    },
  ),
  http.get(`*${PROJECTS_API_PATH}/:projectId`, ({ params }) => {
    const project = getProject(String(params.projectId));
    if (!project) {
      return HttpResponse.json({ error: "not found" }, { status: 404 });
    }
    return HttpResponse.json(project);
  }),
];
