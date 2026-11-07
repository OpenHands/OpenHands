import React from "react";
import { useTranslation } from "react-i18next";
import { isUnencryptedNetworkUrl } from "#/api/mars/mars-device-pairing";
import { I18nKey } from "#/i18n/declaration";

export function UnencryptedConnectionWarning() {
  const { t } = useTranslation("openhands");
  const [visible, setVisible] = React.useState(false);
  React.useEffect(
    () => setVisible(isUnencryptedNetworkUrl(window.location.href)),
    [],
  );
  return visible ? (
    <p
      role="alert"
      className="border-b border-border bg-surface-raised px-4 py-2 text-sm text-feedback-warning"
    >
      {t(I18nKey.BACKEND$UNENCRYPTED_CONNECTION)}
    </p>
  ) : null;
}
