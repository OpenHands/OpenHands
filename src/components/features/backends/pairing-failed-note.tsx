import React from "react";
import { useTranslation } from "react-i18next";

import { hadPairingFailure } from "#/api/mars/pairing";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

/**
 * Shown where a key is asked for, when this tab arrived through a pairing
 * link that no longer works, so the request for a key is not a surprise.
 */
export function PairingFailedNote({
  testId,
  className,
}: {
  testId: string;
  className?: string;
}) {
  const { t } = useTranslation("openhands");
  const [failed] = React.useState(hadPairingFailure);
  if (!failed) return null;
  return (
    <p
      role="alert"
      data-testid={testId}
      className={cn("text-pretty text-sm text-warning", className)}
    >
      {t(I18nKey.AUTH$PAIRING_LINK_FAILED)}
    </p>
  );
}
