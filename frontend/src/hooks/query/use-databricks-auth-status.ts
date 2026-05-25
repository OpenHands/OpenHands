import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import DatabricksAuthService, {
  type DatabricksAuthStatus,
} from "#/api/databricks-auth-service/databricks-auth-service.api";

const QUERY_KEY = ["databricks-auth-status"] as const;

interface UseDatabricksAuthStatusOptions {
  /**
   * Only fire the query when the surrounding component is actually shown
   * (e.g. the user is on the LLM settings tab with provider=databricks).
   * Prevents an unconditional hit on every page load.
   */
  enabled?: boolean;
}

/**
 * TanStack-Query hook that reports whether the browser session holds an
 * active Databricks U2M token and whether U2M is configured in this
 * deployment.
 *
 * The backend route returns ``configured=false`` for deployments that never
 * set ``DATABRICKS_HOST`` + ``DATABRICKS_U2M_CLIENT_ID``; callers should
 * hide the Sign-in affordance entirely in that case rather than showing a
 * button that 501s on click.
 */
export const useDatabricksAuthStatus = (
  options?: UseDatabricksAuthStatusOptions,
) => {
  const { enabled = true } = options ?? {};
  return useQuery<DatabricksAuthStatus>({
    queryKey: QUERY_KEY,
    queryFn: () => DatabricksAuthService.status(),
    // Shorter than the models query — auth state can change any time the
    // user completes the OAuth redirect in another tab. 30s keeps the
    // indicator fresh without hammering the endpoint.
    staleTime: 1000 * 30,
    gcTime: 1000 * 60 * 5,
    retry: false,
    enabled,
  });
};

/**
 * Companion mutation: clears the session's Databricks U2M tokens and
 * invalidates the cached status so the "Sign in" CTA comes back
 * immediately in the UI.
 */
export const useDatabricksLogout = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => DatabricksAuthService.logout(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEY });
    },
  });
};
