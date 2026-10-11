import { useTranslation } from "react-i18next";
import { cn } from "#/utils/utils";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import { I18nKey } from "#/i18n/declaration";
import { ToggleSwitchVisual } from "#/ui/toggle-switch";

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
        "flex items-center gap-2 px-3 py-2 mt-0.5 rounded-lg border cursor-pointer transition-colors min-w-0 max-w-full",
        "has-[:focus-visible]:ring-2 has-[:focus-visible]:ring-focus has-[:focus-visible]:outline-none",
        checked
          ? "border-contrast/30 bg-contrast/5"
          : "border-border bg-transparent hover:bg-contrast/[0.03]",
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
      <ToggleSwitchVisual enabled={checked} />
      <span className="text-sm text-contrast shrink-0">
        {t(I18nKey.MCP$ALSO_SAVE_AS_SECRET)}
      </span>
      <code
        title={fieldKey}
        className={cn(
          "ml-auto text-[11px] font-mono tracking-tight border rounded px-1.5 py-0.5 truncate min-w-0",
          checked
            ? "text-contrast border-contrast/30 bg-contrast/5"
            : "text-contrast border-border",
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
          className="flex items-center justify-center size-3.75 shrink-0 rounded-full border border-contrast/30 text-contrast text-[9px] font-bold cursor-help hover:bg-contrast/10"
          onClick={(e) => e.preventDefault()}
        >
          ?
        </button>
      </StyledTooltip>
    </label>
  );
}
