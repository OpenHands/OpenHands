import { useQuery } from "@tanstack/react-query";
import CanvasExtensionsService from "#/api/canvas-extensions-service";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { CANVAS_EXTENSIONS_QUERY_KEYS } from "#/hooks/query/query-keys";

export function useCanvasExtensionIcon(name: string, enabled: boolean) {
  const active = useActiveBackend();

  return useQuery({
    queryKey: CANVAS_EXTENSIONS_QUERY_KEYS.icon(
      active.backend.id,
      active.orgId,
      active.backend.connectionRevision ?? 0,
      name,
    ),
    queryFn: () => CanvasExtensionsService.fetchIcon(name),
    enabled,
    retry: false,
    meta: { disableToast: true },
  });
}
