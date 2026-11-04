import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";

/**
 * Warning lines for credential pairs that break each other at runtime (see
 * ``getAcpCredentialConflicts``). Rendered identically by the onboarding
 * credentials step and the Settings → Agent credentials section.
 */
// @spec WN-001 — Credential conflict warnings use theme-aware warning feedback
export function AcpConflictWarnings({
  conflicts,
}: {
  conflicts: Array<[string, string]>;
}) {
  const { t } = useTranslation("openhands");

  return (
    <>
      {conflicts.map(([credential, conflicting]) => (
        <p
          key={`${credential}:${conflicting}`}
          data-testid="acp-credential-conflict-warning"
          className="text-sm text-feedback-warning"
        >
          {t(I18nKey.SETTINGS$ACP_CREDENTIAL_CONFLICT_WARNING, {
            credential,
            conflicting,
          })}
        </p>
      ))}
    </>
  );
}
