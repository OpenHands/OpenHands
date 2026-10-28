import React from "react";
import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router";
import { CircleAlert, Copy, Check, ExternalLink } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import type { PluginErrorDetail } from "#/types/plugin-error";

interface PluginErrorBannerProps {
  error: PluginErrorDetail;
  onDismiss?: () => void;
}

/**
 * Specialized error banner for plugin manifest failures.
 *
 * Shows the plugin name, what went wrong, how to fix it, and a copyable
 * error_id for support correlation. Includes a button to navigate to
 * Settings > Skills where the user can disable auto-load.
 */
export function PluginErrorBanner({
  error,
  onDismiss,
}: PluginErrorBannerProps) {
  const { t } = useTranslation("openhands");
  const navigate = useNavigate();
  const [isCopied, setIsCopied] = React.useState(false);

  const handleCopyErrorId = async () => {
    try {
      await navigator.clipboard.writeText(error.error_id);
      setIsCopied(true);
      setTimeout(() => setIsCopied(false), 2000);
    } catch {
      // Clipboard unavailable — ignore
    }
  };

  const handleOpenSettings = () => {
    navigate("/settings/skills");
    onDismiss?.();
  };

  return (
    <div
      className={cn(
        "flex w-full flex-col gap-2 rounded-lg border border-[var(--oh-border)] bg-[var(--oh-surface-raised)] p-3 text-[var(--oh-foreground)]",
      )}
      data-testid="plugin-error-banner"
    >
      {/* Header row */}
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2">
          <CircleAlert
            aria-hidden
            className="h-4 w-4 shrink-0 text-[var(--oh-warning)]"
            strokeWidth={2}
          />
          <span
            className="text-sm font-semibold"
            data-testid="plugin-error-banner-title"
          >
            {t(I18nKey.ERROR$PLUGIN_ERROR_TITLE)}
          </span>
        </div>
        {onDismiss && (
          <button
            type="button"
            onClick={onDismiss}
            className="shrink-0 cursor-pointer rounded-md p-1 text-[var(--oh-muted)] hover:bg-[var(--oh-interactive-hover)] hover:text-[var(--oh-foreground)]"
            aria-label={t(I18nKey.BUTTON$CLOSE)}
            data-testid="plugin-error-banner-dismiss"
          >
            <span aria-hidden className="block h-4 w-4 leading-4 text-center">
              ×
            </span>
          </button>
        )}
      </div>

      {/* Plugin name + error_id row */}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-[var(--oh-muted)]">
        <span data-testid="plugin-error-banner-name">
          <span className="font-medium text-[var(--oh-foreground)]">
            {error.plugin_name}
          </span>
        </span>
        <button
          type="button"
          onClick={handleCopyErrorId}
          className="flex cursor-pointer items-center gap-1 rounded px-1 py-0.5 text-[var(--oh-muted)] hover:bg-[var(--oh-interactive-hover)] hover:text-[var(--oh-foreground)]"
          aria-label={t(isCopied ? I18nKey.BUTTON$COPIED : I18nKey.BUTTON$COPY)}
          data-testid="plugin-error-banner-copy-id"
        >
          <span className="font-mono">{error.error_id.slice(0, 8)}</span>
          {isCopied ? (
            <Check className="h-3 w-3" aria-hidden />
          ) : (
            <Copy className="h-3 w-3" aria-hidden />
          )}
        </button>
      </div>

      {/* Failure reason */}
      <div
        className="text-sm text-[var(--oh-muted)]"
        data-testid="plugin-error-banner-reason"
      >
        {error.failure_reason}
      </div>

      {/* Remediation */}
      <div
        className="text-sm text-[var(--oh-foreground)]"
        data-testid="plugin-error-banner-remediation"
      >
        <span className="font-medium">
          {t(I18nKey.ERROR$PLUGIN_ERROR_REMEDIATION_LABEL)}
        </span>{" "}
        {error.remediation}
      </div>

      {/* Action buttons */}
      <div className="flex items-center gap-2 pt-1">
        <button
          type="button"
          onClick={handleOpenSettings}
          className="flex cursor-pointer items-center gap-1 rounded-md border border-[var(--oh-border)] px-2 py-1 text-xs font-normal text-[var(--oh-foreground)] hover:bg-[var(--oh-interactive-hover)]"
          data-testid="plugin-error-banner-settings"
        >
          <ExternalLink className="h-3 w-3" aria-hidden />
          {t(I18nKey.ERROR$PLUGIN_ERROR_SETTINGS_BUTTON)}
        </button>
      </div>
    </div>
  );
}
