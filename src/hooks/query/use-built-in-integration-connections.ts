import { useQuery } from "@tanstack/react-query";
import {
  fetchCloudJiraLinkStatus,
  fetchCloudSlackStatus,
} from "#/api/cloud/settings-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";

interface BuiltInIntegrationConnectionsOptions {
  /** Whether the instance offers the built-in Slack integration. */
  slackEnabled: boolean;
  /** Whether the instance offers the built-in Jira Cloud integration. */
  jiraEnabled: boolean;
}

/**
 * Whether the current user has connected the cloud instance's built-in Slack
 * and Jira Cloud integrations. Git providers are read from the settings'
 * `provider_tokens_set` instead (see `useNativeGitIntegrations`); these two
 * have lookups of their own on the cloud backend. Each query runs only on a
 * cloud backend whose instance offers the integration, and an answer that is
 * not known yet reads as "not connected".
 */
export function useBuiltInIntegrationConnections({
  slackEnabled,
  jiraEnabled,
}: BuiltInIntegrationConnectionsOptions) {
  const { backend } = useActiveBackend();
  const isCloud = backend.kind === "cloud";
  const revision = backend.connectionRevision ?? 0;

  const { data: isSlackConnected } = useQuery({
    queryKey: ["cloud-slack-status", backend.id, revision],
    queryFn: fetchCloudSlackStatus,
    enabled: isCloud && slackEnabled,
    staleTime: 1000 * 60 * 5,
    retry: false,
    meta: { disableToast: true },
  });
  const { data: isJiraConnected } = useQuery({
    queryKey: ["cloud-jira-link", backend.id, revision],
    queryFn: fetchCloudJiraLinkStatus,
    enabled: isCloud && jiraEnabled,
    staleTime: 1000 * 60 * 5,
    retry: false,
    meta: { disableToast: true },
  });

  return {
    isSlackConnected: isSlackConnected === true,
    isJiraConnected: isJiraConnected === true,
  };
}
