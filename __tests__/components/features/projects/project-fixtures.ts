import { vi } from "vitest";
import ProjectsService, {
  type ProjectArtifact,
  type ProjectConversation,
} from "#/api/projects-service/projects-service.api";
import * as pullRequests from "#/api/git-service/pull-request-status.api";
import { experimentalFeaturesService } from "#/api/experimental-features-service/experimental-features-service.api";

/** One project with a coordinator, three conversations, files and PRs. */
export const PROJECT_ID = "aa000000000040008000000000000001";
export const COORDINATOR = "cc000000000040008000000000000001";
export const RETRY = "bb000000000040008000000000000001";
export const LIMITS = "bb000000000040008000000000000002";
export const AUDIT = "bb000000000040008000000000000003";
const NOW = "2026-11-17T00:00:00Z";

export const member = (
  id: string,
  title: string,
  extra: Partial<ProjectConversation> = {},
): ProjectConversation => ({
  conversation_id: id,
  title,
  execution_status: "finished",
  attention: "idle",
  resolved: null,
  turn: "finished",
  ask: null,
  pull_requests: [],
  automation_id: null,
  parent_conversation_id: COORDINATOR,
  created_at: NOW,
  updated_at: NOW,
  ...extra,
});

export const BOARD = {
  project_id: PROJECT_ID,
  coordinator: member(COORDINATOR, "Login hardening", {
    parent_conversation_id: null,
  }),
  conversations: [
    member(LIMITS, "Rate-limit login", {
      attention: "asking",
      turn: "asking",
      ask: "5 or 10 tries per minute?",
    }),
    member(RETRY, "Fix login retry", {
      attention: "working",
      execution_status: "running",
      pull_requests: [
        { owner: "acme", repo: "app", number: 412 },
        { owner: "acme", repo: "app", number: 409 },
      ],
    }),
    member(AUDIT, "Audit auth flows", {
      attention: "resolved",
      resolved: "user",
      // Also named by "Fix login retry": counted and listed once.
      pull_requests: [
        { owner: "acme", repo: "lib", number: 9 },
        { owner: "acme", repo: "app", number: 412 },
      ],
    }),
  ],
};

const artifact = (
  name: string,
  extra: Partial<ProjectArtifact> = {},
): ProjectArtifact => ({
  name,
  size: 10,
  source: "agent",
  conversation_id: RETRY,
  created_at: NOW,
  updated_at: NOW,
  ...extra,
});

export const ARTIFACTS = [
  artifact("retry-design.md"),
  artifact("backoff.txt"),
  artifact("threat-model.md", { source: "user", conversation_id: null }),
];

/** Answers every Projects call the views make, with the project above. */
export function mockProjectServices() {
  vi.spyOn(
    experimentalFeaturesService,
    "listExperimentalFeatures",
  ).mockResolvedValue([
    { key: "projects", label: "Projects", description: "", enabled: true },
  ] as Awaited<
    ReturnType<typeof experimentalFeaturesService.listExperimentalFeatures>
  >);
  vi.spyOn(ProjectsService, "getProject").mockResolvedValue({
    id: PROJECT_ID,
    name: "Login hardening",
    description: "",
    coordinator_conversation_id: COORDINATOR,
    worker_agent_profile_id: null,
    created_at: NOW,
    updated_at: NOW,
  });
  vi.spyOn(ProjectsService, "listConversations").mockResolvedValue(BOARD);
  vi.spyOn(ProjectsService, "listArtifacts").mockResolvedValue(ARTIFACTS);
  vi.spyOn(ProjectsService, "readArtifactText").mockResolvedValue("text");
  vi.spyOn(ProjectsService, "downloadArtifact").mockResolvedValue(
    new Blob(["text"], { type: "text/plain" }),
  );
  vi.spyOn(pullRequests, "getPullRequestStatus").mockImplementation(
    async (owner, repo, number) => {
      if (number === 409) throw new Error("409 missing_provider_token");
      return {
        repository: `${owner}/${repo}`,
        number,
        title: number === 412 ? "Retry login on network errors" : "Lib fix",
        url: `https://github.com/${owner}/${repo}/pull/${number}`,
        state: number === 412 ? "open" : "merged",
        checks: number === 412 ? "success" : "none",
      };
    },
  );
}
