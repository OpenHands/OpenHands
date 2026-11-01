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

  // @spec MCP-005 — Credential selection stays readable in every theme
  return (
    <label
      data-testid={`mcp-install-save-secret-${fieldKey}`}
      className={cn(
        "flex flex-wrap items-center gap-2 px-3 py-2 mt-0.5 rounded-lg border cursor-pointer transition-colors",
        "has-focus-visible:outline-2 has-focus-visible:outline-offset-2 has-focus-visible:outline-focus",
        checked
          ? "border-contrast/25 bg-contrast/5"
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
      <span className="text-sm text-contrast">
        {t(I18nKey.MCP$ALSO_SAVE_AS_SECRET)}
      </span>
      <span className="ml-auto flex min-w-0 max-w-full items-center gap-2">
        <code
          className={cn(
            "min-w-0 break-all text-[11px] text-contrast font-mono tracking-tight border rounded px-1.5 py-0.5",
            checked ? "border-contrast/25 bg-contrast/4" : "border-border",
          )}
        >
          {fieldKey}
        </code>
        <StyledTooltip
          content={t(I18nKey.MCP$SAVE_AS_SECRET_TOOLTIP)}
          placement="top"
        >
          {/* The help button must stay keyboard-reachable without toggling. */}
          <button
            type="button"
            aria-label={t(I18nKey.MCP$SAVE_AS_SECRET_TOOLTIP)}
            className="flex items-center justify-center size-3.75 shrink-0 rounded-full border border-muted text-contrast text-[9px] font-bold cursor-help"
            onClick={(e) => e.preventDefault()}
          >
            ?
          </button>
        </StyledTooltip>
      </span>
    </label>
  );
}
