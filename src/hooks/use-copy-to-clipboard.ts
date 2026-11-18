import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { copyToClipboard } from "#/utils/copy-to-clipboard";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

/**
 * Returns a `copy(text)` function that copies to the clipboard and shows one
 * shared error toast when it fails. Resolves to whether the copy worked, so
 * callers only handle success (for example, a "copied" checkmark).
 */
export function useCopyToClipboard() {
  const { t } = useTranslation("openhands");

  return React.useCallback(
    async (text: string) => {
      const copied = await copyToClipboard(text);
      if (!copied) {
        displayErrorToast(t(I18nKey.COMMON$COPY_FAILED));
      }
      return copied;
    },
    [t],
  );
}
