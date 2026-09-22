import React from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { useTranslation } from "react-i18next";
import { AcpSecretField } from "#/components/features/settings/acp-secret-field";
import { I18nKey } from "#/i18n/declaration";
import type { AcpCredentialForm } from "#/hooks/use-acp-credential-form";

interface AcpCredentialFieldsProps {
  form: AcpCredentialForm;
  /**
   * Whether the host-login probe confirmed a session for the provider. When it
   * did, the fields are redundant — the agent authenticates through that login
   * — so they collapse behind the "Advanced" toggle (see #16296).
   */
  isAuthenticated: boolean;
  /**
   * Prefix for the field and toggle test ids, e.g. ``"onboarding-acp"`` →
   * ``onboarding-acp-secret-ANTHROPIC_API_KEY`` /
   * ``onboarding-acp-advanced-toggle``.
   */
  testIdPrefix: string;
}

/**
 * The provider's credential inputs, shared by the ACP onboarding step and
 * Settings → Agent.
 *
 * When the login probe reports a signed-in host session the green banner above
 * already says the agent can authenticate, so leaving a column of empty API-key
 * fields below it reads as "something is still missing" (#16296). They collapse
 * behind an "Advanced" toggle instead — still one click away for anyone adding
 * or rotating a key, and still expanded whenever the fields are the only way to
 * authenticate, or a conflict warning is pointing at them.
 */
export function AcpCredentialFields({
  form,
  isAuthenticated,
  testIdPrefix,
}: AcpCredentialFieldsProps) {
  const { t } = useTranslation("openhands");
  const [expanded, setExpanded] = React.useState(false);
  const fieldsId = React.useId();
  const { fields, values, setValue, secretExists, conflicts } = form;

  // A conflict warning names two of these fields, so hiding them would leave
  // the user nothing to act on.
  const collapsible = isAuthenticated && conflicts.length === 0;
  const Chevron = expanded ? ChevronDown : ChevronRight;

  const list = (
    <div id={fieldsId} className="flex flex-col gap-5">
      {fields.map((field) => (
        <AcpSecretField
          key={field.name}
          field={field}
          value={values[field.name] ?? ""}
          onChange={(value) => setValue(field.name, value)}
          alreadySet={secretExists(field.name)}
          testId={`${testIdPrefix}-secret-${field.name}`}
          showOptionalTag
        />
      ))}
    </div>
  );

  if (!collapsible) return list;

  return (
    <div className="flex flex-col gap-5">
      <button
        type="button"
        onClick={() => setExpanded((prev) => !prev)}
        aria-expanded={expanded}
        aria-controls={fieldsId}
        data-testid={`${testIdPrefix}-advanced-toggle`}
        className="flex w-fit cursor-pointer items-center gap-1.5 text-sm text-muted hover:text-contrast"
      >
        <Chevron className="size-4 shrink-0" aria-hidden />
        <span>{t(I18nKey.SETTINGS$ACP_ADVANCED_CREDENTIALS)}</span>
      </button>
      {expanded && list}
    </div>
  );
}
