import type {
  GitChangeStatus,
  AgentServerGitChangeStatus,
} from "#/api/open-hands.types";

type ClientGitChangeStatus = "added" | "modified" | "deleted" | "renamed";

type SupportedGitStatus = AgentServerGitChangeStatus | ClientGitChangeStatus;

export function mapAnyGitStatusToClientStatus(
  status: SupportedGitStatus,
): GitChangeStatus {
  switch (status) {
    case "ADDED":
    case "added":
      return "A";
    case "DELETED":
    case "deleted":
      return "D";
    case "UPDATED":
    case "modified":
      return "M";
    case "MOVED":
    case "renamed":
      return "R";
    default:
      return "M";
  }
}

export function mapAgentServerToClientGitStatus(
  agentServerStatus: AgentServerGitChangeStatus,
): GitChangeStatus {
  return mapAnyGitStatusToClientStatus(agentServerStatus);
}
