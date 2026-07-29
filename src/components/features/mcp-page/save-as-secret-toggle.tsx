import { useTranslation } from "react-i18next";
import { cn } from "#/utils/utils";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import { I18nKey } from "#/i18n/declaration";

interface SaveAsSecretToggleProps {
  fieldKey: string;
  checked: boolean;
  onToggle: (value: boolean) => void;
}

export function SaveAsSecretToggle({
  fieldKey,
  checked,
  onToggle,
}: SaveAsSecretToggleProps) {
  const { t } = useTranslation("openhands");

  return (
    <label
      data-testid={`mcp-install-save-secret-${fieldKey}`}
      className={cn(
        "flex items-center gap-2 px-3 py-2 mt-0.5 rounded-lg border cursor-pointer transition-colors",
        checked
          ? "border-green-500/35 bg-green-500/10"
          : "border-[var(--oh-border)] bg-transparent hover:bg-white/[0.03]",
      )}
    >
      {/* sr-only keeps the real checkbox in the accessibility tree so AT
          users can toggle it without seeing the custom visual track. */}
      <input
        className="sr-only"
        id={`mcp-save-secret-checkbox-${fieldKey}`}
        type="checkbox"
        checked={checked}
        onChange={(e) => onToggle(e.target.checked)}
      />
      {/* aria-hidden: purely decorative — the checkbox above is the semantic control. */}
      <span
        aria-hidden="true"
        className={cn(
          "relative inline-flex h-[22px] w-[40px] shrink-0 items-center rounded-full border transition-colors duration-200",
          checked
            ? "border-green-500 bg-green-500"
            : "border-[var(--oh-border)] bg-surface-raised",
        )}
      >
        <span
          className={cn(
            "inline-block size-4 rounded-full transition-transform duration-200",
            checked
              ? "translate-x-[21px] bg-white"
              : "translate-x-[2px] bg-[var(--oh-muted)]",
          )}
        />
      </span>
      <span className="text-sm">{t(I18nKey.MCP$ALSO_SAVE_AS_SECRET)}</span>
      <code
        className={cn(
          "ml-auto text-[11px] font-mono tracking-tight border rounded px-1.5 py-0.5",
          checked
            ? "text-green-500 border-green-500/35 bg-white/[0.04]"
            : "text-tertiary-alt border-[var(--oh-border)]",
        )}
      >
        {fieldKey}
      </code>
      <StyledTooltip
        content={t(I18nKey.MCP$SAVE_AS_SECRET_TOOLTIP)}
        placement="top"
      >
        {/* button so the tooltip is keyboard-reachable; type=button prevents
            accidental form submission when the user presses Enter. */}
        <button
          type="button"
          aria-label={t(I18nKey.MCP$SAVE_AS_SECRET_TOOLTIP)}
          className="flex items-center justify-center size-[15px] shrink-0 rounded-full border border-[var(--oh-muted)] text-tertiary-alt text-[9px] font-bold cursor-help"
          onClick={(e) => e.preventDefault()}
        >
          ?
        </button>
      </StyledTooltip>
    </label>
  );
}
