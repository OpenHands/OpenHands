import { ExternalLink } from "lucide-react";
import { useTranslation } from "react-i18next";
import VSCodeIcon from "#/icons/vscode.svg?react";
import { I18nKey } from "#/i18n/declaration";
import { useAgentState } from "#/hooks/use-agent-state";
import { useUnifiedVSCodeUrl } from "#/hooks/query/use-unified-vscode-url";
import { RUNTIME_STARTING_STATES } from "#/types/agent-state";
import { cn } from "#/utils/utils";

export function DrawerVSCodeLink() {
  const { t } = useTranslation("openhands");
  const { curAgentState } = useAgentState();
  const { data, refetch, isLoading } = useUnifiedVSCodeUrl();
  const isRuntimeStarting = RUNTIME_STARTING_STATES.includes(curAgentState);

  const handleClick = async () => {
    let vscodeUrl = data?.url;

    if (!vscodeUrl) {
      const result = await refetch();
      vscodeUrl = result.data?.url ?? null;
    }

    if (vscodeUrl) {
      window.open(vscodeUrl, "_blank", "noopener,noreferrer");
    }
  };

  return (
    <button
      type="button"
      onClick={handleClick}
      disabled={isLoading || isRuntimeStarting}
      aria-label={t(I18nKey.VSCODE$OPEN)}
      title={t(I18nKey.VSCODE$OPEN)}
      data-testid="drawer-vscode-link"
      className={cn(
        "inline-flex items-center gap-1.5 rounded-[7px] border border-[var(--oh-border)] bg-base-secondary px-2 py-1 text-xs",
        "text-[var(--oh-muted)] transition-colors hover:enabled:bg-surface-raised hover:enabled:text-white",
        "cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed",
      )}
    >
      <VSCodeIcon className="h-[15px] w-[15px] shrink-0" aria-hidden />
      <span>{t(I18nKey.FILES$VSCODE)}</span>
      <ExternalLink
        className="h-3.5 w-3.5 shrink-0"
        aria-hidden
        strokeWidth={2}
      />
    </button>
  );
}
