import { useQuery } from "@tanstack/react-query";
import AutomationService from "#/api/automation-service/automation-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import type { Automation } from "#/types/automation";
import { unpackTarGzip } from "#/utils/tar-unpack";

export const AUTOMATION_TARBALL_FILES_QUERY_KEY = [
  "automation-tarball-files",
] as const;

/** Bundles with more regular-file content than this are unavailable. */
const MAX_CONTENT_BYTES = 16 * 1024 * 1024;
/** Allow tar headers and padding without changing the content limit. */
const MAX_ARCHIVE_BYTES = MAX_CONTENT_BYTES + 1024 * 1024;

/**
 * The files inside an automation's uploaded bundle, read back from the
 * service's tarball endpoint so the detail page can show what a script
 * automation runs. `updated_at` is part of the key: an edit that replaces
 * the bundle changes it, so the stale unpack is not reused.
 *
 * Errors are the section's to render (the endpoint answers 404 for a
 * deleted upload and 422 for an externally hosted bundle), so the global
 * toast is disabled.
 */
export function useAutomationTarballFiles(
  automation: Pick<Automation, "id" | "updated_at">,
) {
  const active = useActiveBackend();
  return useQuery({
    queryKey: [
      ...AUTOMATION_TARBALL_FILES_QUERY_KEY,
      automation.id,
      automation.updated_at,
      active.backend.id,
      active.orgId,
    ],
    queryFn: async () =>
      unpackTarGzip(await AutomationService.fetchTarballBytes(automation.id), {
        maxContentBytes: MAX_CONTENT_BYTES,
        maxArchiveBytes: MAX_ARCHIVE_BYTES,
      }),
    staleTime: 5 * 60 * 1000,
    retry: false,
    enabled: !!automation.id,
    meta: { disableToast: true },
  });
}
