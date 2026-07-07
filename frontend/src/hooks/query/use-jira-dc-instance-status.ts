import { useQuery } from "@tanstack/react-query";
import { openHands } from "#/api/open-hands-axios";

export interface JiraDcInstanceStatus {
  /** Whether an admin has set up the install's Jira Data Center connection. */
  configured: boolean;
  /** Host to link to (drives the per-user link call); null when not configured. */
  host: string | null;
}

/**
 * Instance-level Jira DC status: whether the connection is set up and its host.
 * Lets a not-yet-linked member see "link your account" vs "ask an admin to set
 * it up". Only needed for a member in email mode — gate it via `enabled`.
 */
export function useJiraDcInstanceStatus(enabled = true) {
  return useQuery<JiraDcInstanceStatus>({
    queryKey: ["jira-dc-instance-status"],
    enabled,
    queryFn: async () => {
      const response = await openHands.get(
        "/integration/jira-dc/workspaces/status",
      );
      return response.data;
    },
  });
}
