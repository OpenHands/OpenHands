/**
 * The live state of one GitHub pull request, from the Agent Server's
 * `GET /api/git/repositories/{owner}/{repo}/pull-requests/{number}`, which
 * reads it with the user's provider token. Without a token it answers 409
 * with `{"code": "missing_provider_token"}`, and callers show the link alone.
 */
import { AgentServerClient } from "@openhands/typescript-client/clients";
import { getAgentServerClientOptions } from "../agent-server-client-options";

export interface PullRequestStatus {
  repository: string;
  number: number;
  title: string;
  url: string;
  state: "open" | "closed" | "merged";
  draft?: boolean;
  /** The combined CI result of the head commit. */
  checks: "success" | "failure" | "pending" | "none";
}

export async function getPullRequestStatus(
  owner: string,
  repo: string,
  number: number,
): Promise<PullRequestStatus> {
  return new AgentServerClient(
    getAgentServerClientOptions(),
  ).request<PullRequestStatus>({
    method: "GET",
    path: `/api/git/repositories/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}/pull-requests/${number}`,
    params: { provider: "github" },
  });
}
