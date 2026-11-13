import { useMutation } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import type { WorkspaceFileContent } from "#/hooks/query/use-workspace-file-content";
import { I18nKey } from "#/i18n/declaration";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { downloadBlob } from "#/utils/utils";

async function fetchWorkspaceFile(file: WorkspaceFileContent): Promise<Blob> {
  // Same cookie-authenticated request the viewer makes; `no-cache` makes the
  // browser revalidate so Refresh and agent edits are reflected (#17921).
  const response = await fetch(file.staticUrl, {
    credentials: "include",
    cache: "no-cache",
  });
  if (!response.ok) {
    throw new Error(`Failed to download ${file.path}: ${response.status}`);
  }
  return response.blob();
}

/**
 * Save a Files tab file to the user's machine under its own file name.
 *
 * Local backends fetch the bytes from the workspace fileserver `staticUrl`,
 * so every kind (text, image, PDF, binary) downloads byte-for-byte. Cloud
 * backends only hand the file back as UTF-8 text through
 * `GET /api/v1/app-conversations/{id}/file` (the `data:` `staticUrl`), so the
 * blob is rebuilt from that text. Cloud binaries wait on OHE-3631; callers
 * should not offer the download for a Cloud file without decoded text.
 */
export function useDownloadWorkspaceFile() {
  const { t } = useTranslation("openhands");

  return useMutation({
    mutationFn: async (file: WorkspaceFileContent) => {
      const blob = file.staticUrl.startsWith("data:")
        ? new Blob([file.text ?? ""], {
            type: `${file.mimeType};charset=utf-8`,
          })
        : await fetchWorkspaceFile(file);
      downloadBlob(blob, file.path.split("/").pop() || file.path);
    },
    // The toast below is the only one for a failed download; without this the
    // global MutationCache handler adds a second, raw error toast.
    meta: { disableToast: true },
    onError: () => displayErrorToast(t(I18nKey.FILES$DOWNLOAD_ERROR)),
  });
}
