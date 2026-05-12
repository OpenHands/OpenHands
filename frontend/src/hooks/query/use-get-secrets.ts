import {
  useQuery,
  useInfiniteQuery,
  InfiniteData,
} from "@tanstack/react-query";
import { SecretsService } from "#/api/secrets-service";
import { useConfig } from "./use-config";
import { useIsAuthed } from "#/hooks/query/use-is-authed";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";
import { CustomSecretPage } from "#/api/secrets-service.types";

/**
 * @deprecated Use useSearchSecrets instead for paginated access
 */
export const useGetSecrets = () => {
  const { data: config } = useConfig();
  const { data: isAuthed } = useIsAuthed();
  const { organizationId } = useSelectedOrganizationId();

  const isOss = config?.app_mode === "oss";

  return useQuery({
    queryKey: ["secrets", organizationId],
    queryFn: SecretsService.getSecrets,
    enabled: isOss || (isAuthed && !!organizationId),
  });
};

interface UseSearchSecretsOptions {
  nameContains?: string;
  pageSize?: number;
  enabled?: boolean;
}

/**
 * Hook for searching/listing secrets with infinite scroll pagination support.
 */
export const useSearchSecrets = (options: UseSearchSecretsOptions = {}) => {
  const { nameContains, pageSize = 30, enabled = true } = options;
  const { data: config } = useConfig();
  const { data: isAuthed } = useIsAuthed();
  const { organizationId } = useSelectedOrganizationId();

  const isOss = config?.app_mode === "oss";
  const isEnabled = enabled && (isOss || (isAuthed && !!organizationId));

  const query = useInfiniteQuery<
    CustomSecretPage,
    Error,
    InfiniteData<CustomSecretPage>,
    [string, string | null, string | undefined, number],
    string | null
  >({
    queryKey: ["secrets-search", organizationId, nameContains, pageSize],
    queryFn: async ({ pageParam }) =>
      SecretsService.searchSecrets({
        name__contains: nameContains,
        page_id: pageParam ?? undefined,
        limit: pageSize,
      }),
    getNextPageParam: (lastPage) => lastPage.next_page_id,
    initialPageParam: null,
    enabled: isEnabled,
    staleTime: 1000 * 60 * 5, // 5 minutes
    gcTime: 1000 * 60 * 15, // 15 minutes
  });

  const onLoadMore = () => {
    if (query.hasNextPage && !query.isFetchingNextPage) {
      query.fetchNextPage();
    }
  };

  // Flatten all pages into a single array of secrets
  const secrets = query.data?.pages?.flatMap((page) => page.items) ?? [];

  return {
    data: secrets,
    isLoading: query.isLoading,
    isError: query.isError,
    hasNextPage: query.hasNextPage ?? false,
    isFetchingNextPage: query.isFetchingNextPage,
    fetchNextPage: query.fetchNextPage,
    onLoadMore,
    refetch: query.refetch,
  };
};
