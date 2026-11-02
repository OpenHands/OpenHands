import { useActiveBackend } from "#/contexts/active-backend-context";
import { useProviderModels } from "#/hooks/query/use-provider-models";

export function useModelCatalogWarning() {
  const { backend } = useActiveBackend();
  const isCloud = backend.kind === "cloud";
  const models = useProviderModels(isCloud ? "openhands" : null);

  return (model: string | null | undefined): boolean => {
    // Only a complete, successful catalog can establish that a route is absent.
    if (
      !isCloud ||
      !model?.startsWith("openhands/") ||
      !models.isSuccess ||
      models.isFetching ||
      models.data.length === 0
    ) {
      return false;
    }
    const route = model.slice("openhands/".length);
    return !models.data.some((entry) => entry.name === route);
  };
}
