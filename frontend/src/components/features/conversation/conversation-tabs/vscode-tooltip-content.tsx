import { FaExternalLinkAlt } from "react-icons/fa";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { RUNTIME_INACTIVE_STATES } from "#/types/agent-state";
import { useAgentState } from "#/hooks/use-agent-state";
import { useUnifiedVSCodeUrl } from "#/hooks/query/use-unified-vscode-url";

export function VSCodeTooltipContent() {
  const { curAgentState } = useAgentState();
  const { t } = useTranslation();
  const { data, refetch } = useUnifiedVSCodeUrl();

  const handleVSCodeClick = async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();

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
    <div className="flex items-center gap-2">
      <span>{t(I18nKey.COMMON$CODE)}</span>
      {!RUNTIME_INACTIVE_STATES.includes(curAgentState) ? (
        <FaExternalLinkAlt
          className="w-3 h-3 text-inherit cursor-pointer"
          onClick={handleVSCodeClick}
        />
      ) : null}
    </div>
  );
}
