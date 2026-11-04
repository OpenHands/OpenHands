import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import GitService from "#/api/git-service/git-service.api";
import { useSettings } from "#/hooks/query/use-settings";
import { HomeRepositorySelection, useHomeStore } from "#/stores/home-store";
import { convertRawProvidersToList } from "#/utils/convert-raw-providers-to-list";

// Comfortably above the branch dropdown's page size, so a short exact name
// (e.g. `main`) is not crowded out by longer names that contain it.
const BRANCH_LOOKUP_LIMIT = 100;

const isSelectionShaped = (
  value: unknown,
): value is HomeRepositorySelection => {
  const selection = value as Partial<HomeRepositorySelection> | undefined;
  return (
    typeof selection?.repository?.full_name === "string" &&
    typeof selection?.branch?.name === "string" &&
    typeof selection?.provider === "string"
  );
};

/**
 * The home launcher's repository selection for `backendId`, persisted across
 * visits. A remembered selection is shown immediately and checked in the
 * background; if its provider is disconnected, or the repository or branch
 * can no longer be reached, it is forgotten.
 */
export function useRememberedRepositorySelection(
  backendId: string,
  enabled: boolean,
) {
  const stored = useHomeStore((state) =>
    enabled ? state.repositorySelections[backendId] : undefined,
  );
  const setRepositorySelection = useHomeStore(
    (state) => state.setRepositorySelection,
  );
  const clearRepositorySelection = useHomeStore(
    (state) => state.clearRepositorySelection,
  );
  const { data: settings, isSuccess: hasSettings } = useSettings();

  const selection = isSelectionShaped(stored) ? stored : null;
  const isProviderConnected =
    !!selection &&
    convertRawProvidersToList(settings?.provider_tokens_set).includes(
      selection.provider,
    );
  const repositoryName = selection?.repository.full_name ?? null;
  const branchName = selection?.branch.name ?? null;

  const { data: isReachable, isError } = useQuery({
    queryKey: [
      "repository",
      repositoryName,
      "branches",
      "remembered-selection",
      branchName,
      selection?.provider,
      backendId,
    ],
    queryFn: async () => {
      const page = await GitService.searchRepositoryBranches(
        repositoryName!,
        selection!.provider,
        branchName!,
        undefined,
        BRANCH_LOOKUP_LIMIT,
      );
      return page.items.some((branch) => branch.name === branchName);
    },
    enabled: isProviderConnected,
    staleTime: 1000 * 60 * 5,
    meta: { disableToast: true },
  });

  const isInvalid =
    stored !== undefined &&
    (!selection ||
      (hasSettings && !isProviderConnected) ||
      isError ||
      isReachable === false);

  useEffect(() => {
    if (isInvalid) clearRepositorySelection(backendId);
  }, [isInvalid, backendId, clearRepositorySelection]);

  return {
    selection: isInvalid ? null : selection,
    setSelection: (next: HomeRepositorySelection) =>
      setRepositorySelection(backendId, next),
    clearSelection: () => clearRepositorySelection(backendId),
  };
}
