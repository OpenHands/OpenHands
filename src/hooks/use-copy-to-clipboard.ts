import { useCallback } from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { copyToClipboard } from "#/utils/copy-to-clipboard";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

/**
 * Hook providing a clipboard copy function that displays a shared error toast
 * when copying fails. Returns whether the text reached the clipboard.
 */
export function useCopyToClipboard(): (text: string) => Promise<boolean> {
  const { t } = useTranslation();

  return useCallback(
    async (text: string): Promise<boolean> => {
      const ok = await copyToClipboard(text);
      if (!ok) {
        displayErrorToast(t(I18nKey.CLIPBOARD$COPY_FAILED));
      }
      return ok;
    },
    [t],
  );
}
