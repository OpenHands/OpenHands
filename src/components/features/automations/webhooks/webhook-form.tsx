import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { useCreateWebhook, useUpdateWebhook } from "#/hooks/query/use-webhooks";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { BrandButton } from "#/components/features/settings/brand-button";
import { OptionalTag } from "#/components/features/settings/optional-tag";
import { formControlSettingsFieldClassName } from "#/utils/form-control-classes";
import { cn } from "#/utils/utils";
import type { CustomWebhook } from "#/types/webhook";
import { WebhookSecretReveal } from "./webhook-secret-reveal";

interface WebhookFormProps {
  mode: "add" | "edit";
  webhook: CustomWebhook | null;
  onCancel: () => void;
  onDone: () => void;
}

const SOURCE_PATTERN = "^[a-z0-9][a-z0-9-]{0,48}[a-z0-9]$|^[a-z0-9]$";

export function WebhookForm({
  mode,
  webhook,
  onCancel,
  onDone,
}: WebhookFormProps) {
  const { t } = useTranslation("openhands");
  const { mutate: createWebhook, isPending: isCreating } = useCreateWebhook();
  const { mutate: updateWebhook, isPending: isUpdating } = useUpdateWebhook();

  const [error, setError] = React.useState<string | null>(null);
  const [revealedSecret, setRevealedSecret] = React.useState<string | null>(
    null,
  );

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);

    const formData = new FormData(event.currentTarget);
    const name = formData.get("webhook-name")?.toString().trim();
    const source = formData.get("webhook-source")?.toString().trim();
    const eventKeyExpr =
      formData.get("webhook-event-key-expr")?.toString().trim() || undefined;
    const signatureHeader =
      formData.get("webhook-signature-header")?.toString().trim() || undefined;
    const secret =
      formData.get("webhook-secret")?.toString().trim() || undefined;

    if (!name) return;

    if (mode === "add") {
      if (!source) {
        setError(t(I18nKey.AUTOMATIONS$WEBHOOKS$SOURCE_REQUIRED));
        return;
      }
      createWebhook(
        {
          name,
          source,
          ...(eventKeyExpr && { event_key_expr: eventKeyExpr }),
          ...(signatureHeader && { signature_header: signatureHeader }),
          ...(secret && { webhook_secret: secret }),
        },
        {
          onSuccess: (created) => {
            if (created.webhook_secret) {
              setRevealedSecret(created.webhook_secret);
            } else {
              onDone();
            }
          },
          onError: () => setError(t(I18nKey.AUTOMATIONS$WEBHOOKS$SAVE_ERROR)),
        },
      );
      return;
    }

    if (webhook) {
      updateWebhook(
        {
          id: webhook.id,
          body: {
            name,
            ...(eventKeyExpr && { event_key_expr: eventKeyExpr }),
            ...(signatureHeader && { signature_header: signatureHeader }),
          },
        },
        {
          onSuccess: onDone,
          onError: () => setError(t(I18nKey.AUTOMATIONS$WEBHOOKS$SAVE_ERROR)),
        },
      );
    }
  };

  if (revealedSecret) {
    return <WebhookSecretReveal secret={revealedSecret} onDismiss={onDone} />;
  }

  const formTestId = mode === "add" ? "add-webhook-form" : "edit-webhook-form";
  const isPending = isCreating || isUpdating;

  return (
    <form
      data-testid={formTestId}
      onSubmit={handleSubmit}
      className="flex flex-col items-start gap-6"
    >
      <SettingsInput
        testId="webhook-name-input"
        name="webhook-name"
        type="text"
        label={t(I18nKey.SETTINGS$NAME)}
        className="w-full min-w-0"
        required
        defaultValue={webhook?.name ?? ""}
      />

      {mode === "add" && (
        <SettingsInput
          testId="webhook-source-input"
          name="webhook-source"
          type="text"
          label={t(I18nKey.AUTOMATIONS$WEBHOOKS$SOURCE)}
          className="w-full min-w-0"
          required
          placeholder={t(I18nKey.AUTOMATIONS$WEBHOOKS$SOURCE_PLACEHOLDER)}
          pattern={SOURCE_PATTERN}
          title={t(I18nKey.AUTOMATIONS$WEBHOOKS$SOURCE_PATTERN_TITLE)}
        />
      )}
      {mode === "edit" && webhook && (
        <div className="flex w-full min-w-0 flex-col gap-2.5">
          <span className="text-sm">
            {t(I18nKey.AUTOMATIONS$WEBHOOKS$SOURCE)}
          </span>
          <code className="rounded border border-[var(--oh-border)] bg-[var(--oh-surface-raised)] px-2.5 py-2 font-mono text-sm text-muted">
            {webhook.source}
          </code>
          <span className="text-xs text-muted">
            {t(I18nKey.AUTOMATIONS$WEBHOOKS$SOURCE_IMMUTABLE)}
          </span>
        </div>
      )}

      <SettingsInput
        testId="webhook-event-key-expr-input"
        name="webhook-event-key-expr"
        type="text"
        label={t(I18nKey.AUTOMATIONS$WEBHOOKS$EVENT_KEY_EXPR)}
        className="w-full min-w-0"
        defaultValue={webhook?.event_key_expr ?? ""}
        placeholder={t(I18nKey.AUTOMATIONS$WEBHOOKS$EVENT_KEY_EXPR_PLACEHOLDER)}
      />

      <SettingsInput
        testId="webhook-signature-header-input"
        name="webhook-signature-header"
        type="text"
        label={t(I18nKey.AUTOMATIONS$WEBHOOKS$SIGNATURE_HEADER)}
        className="w-full min-w-0"
        defaultValue={webhook?.signature_header ?? ""}
        placeholder={t(
          I18nKey.AUTOMATIONS$WEBHOOKS$SIGNATURE_HEADER_PLACEHOLDER,
        )}
      />

      {mode === "add" && (
        <label className="flex w-full min-w-0 flex-col gap-2.5">
          <div className="flex items-center gap-2">
            <span className="text-sm">
              {t(I18nKey.AUTOMATIONS$WEBHOOKS$SECRET)}
            </span>
            <OptionalTag />
          </div>
          <input
            data-testid="webhook-secret-input"
            name="webhook-secret"
            type="text"
            className={cn(formControlSettingsFieldClassName, "font-mono")}
            placeholder={t(I18nKey.AUTOMATIONS$WEBHOOKS$SECRET_PLACEHOLDER)}
          />
          <span className="text-xs text-muted">
            {t(I18nKey.AUTOMATIONS$WEBHOOKS$SECRET_HINT)}
          </span>
        </label>
      )}

      {error && <p className="text-sm text-red-500">{error}</p>}

      <div className="flex items-center gap-4">
        <BrandButton
          testId="webhook-cancel-button"
          type="button"
          variant="secondary"
          onClick={onCancel}
        >
          {t(I18nKey.BUTTON$CANCEL)}
        </BrandButton>
        <BrandButton
          testId="webhook-submit-button"
          type="submit"
          variant="primary"
          isDisabled={isPending}
        >
          {mode === "add"
            ? t(I18nKey.AUTOMATIONS$WEBHOOKS$ADD)
            : t(I18nKey.AUTOMATIONS$WEBHOOKS$SAVE)}
        </BrandButton>
      </div>
    </form>
  );
}
