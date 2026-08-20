import { useQuery } from "@tanstack/react-query";
import { useIsAuthed } from "./use-is-authed";
import GitService from "#/api/git-service/git-service.api";
import { useUserProviders } from "../use-user-providers";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { Provider } from "#/types/settings";
import { shouldUseInstallationRepos } from "#/utils/utils";

/**
 * Get the first page of app installations for the provider given.
 *
 * The query key includes the active backend identity so switching
 * between backends (Local ↔ Cloud, Cloud A ↔ Cloud B) naturally produces
 * a fresh query — no `clear()`/`invalidate` orchestration required.
 */
export const useAppInstallations = (selectedProvider: Provider | null) => {
  const { data: userIsAuthenticated } = useIsAuthed();
  const { providers } = useUserProviders();
  const active = useActiveBackend();

  return useQuery({
    // Key on `selectedProvider` + active backend identity. The
    // `providers` array used to be in the key, but it caused a
    // cascading refetch the moment settings resolved and providers
    // flipped from [] → [github] — duplicate "wave 2" with no
    // semantic value (selectedProvider already pins the cache slot).
    queryKey: [
      "installations",
      selectedProvider,
      active.backend.id,
      active.orgId,
    ],
    queryFn: () => GitService.getUserInstallations(selectedProvider!),
    enabled:
      userIsAuthenticated &&
      !!selectedProvider &&
      // Gate on providers length too: when settings haven't yet told
      // us which providers the user has connected we must not fire,
      // otherwise we'd hit the cloud backend for a provider it can't service.
      providers.length > 0 &&
      shouldUseInstallationRepos(selectedProvider, active.backend.kind),
    staleTime: 1000 * 60 * 5, // 5 minutes
    gcTime: 1000 * 60 * 15, // 15 minutes
  });
};
