import { useState } from "react";
import { useTranslation } from "react-i18next";
import { Check, Copy } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";

interface WebhookSecretRevealProps {
  secret: string;
  onDismiss: () => void;
}

/**
 * One-time secret display shown right after a webhook is created or its
 * secret is rotated. The backend never echoes a secret back afterward, so
 * this is the only chance the user has to copy it.
 */
export function WebhookSecretReveal({
  secret,
  onDismiss,
}: WebhookSecretRevealProps) {
  const { t } = useTranslation("openhands");
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(secret);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div
      data-testid="webhook-secret-reveal"
      className="flex flex-col gap-2 rounded-lg border border-[var(--oh-warning-border,theme(colors.yellow.500))] bg-[var(--oh-surface-raised)] p-4"
    >
      <p className="text-sm font-medium text-content">
        {t(I18nKey.AUTOMATIONS$WEBHOOKS$SECRET_REVEAL_TITLE)}
      </p>
      <p className="text-xs text-muted">
        {t(I18nKey.AUTOMATIONS$WEBHOOKS$SECRET_REVEAL_HINT)}
      </p>
      <div className="flex items-center gap-2">
        <code
          data-testid="webhook-secret-value"
          className="min-w-0 flex-1 truncate rounded border border-[var(--oh-border)] bg-[var(--oh-surface)] px-2 py-1.5 font-mono text-xs text-content"
        >
          {secret}
        </code>
        <button
          type="button"
          data-testid="copy-webhook-secret-button"
          onClick={handleCopy}
          aria-label={t(I18nKey.AUTOMATIONS$WEBHOOKS$COPY_SECRET)}
          className="flex shrink-0 items-center gap-1 rounded border border-[var(--oh-border)] px-2 py-1.5 text-xs text-content hover:bg-[var(--oh-surface)]"
        >
          {copied ? (
            <Check aria-hidden className="size-3.5" />
          ) : (
            <Copy aria-hidden className="size-3.5" />
          )}
          {copied
            ? t(I18nKey.AUTOMATIONS$WEBHOOKS$COPIED)
            : t(I18nKey.AUTOMATIONS$WEBHOOKS$COPY_SECRET)}
        </button>
      </div>
      <button
        type="button"
        data-testid="dismiss-webhook-secret-button"
        onClick={onDismiss}
        className="self-start text-xs text-muted underline hover:text-content-2"
      >
        {t(I18nKey.AUTOMATIONS$WEBHOOKS$SECRET_REVEAL_DISMISS)}
      </button>
    </div>
  );
}
