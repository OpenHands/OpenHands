import { FaExternalLinkAlt } from "react-icons/fa";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { useAgentState } from "#/hooks/use-agent-state";
import { useUnifiedVSCodeUrl } from "#/hooks/query/use-unified-vscode-url";
import { RUNTIME_STARTING_STATES } from "#/types/agent-state";

export function VSCodeTooltipContent() {
  const { curAgentState } = useAgentState();
  const { t } = useTranslation();
  const { data, refetch } = useUnifiedVSCodeUrl();
  const isRuntimeStarting = RUNTIME_STARTING_STATES.includes(curAgentState);

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
      {!isRuntimeStarting ? (
        <FaExternalLinkAlt
          className="w-3 h-3 text-inherit cursor-pointer"
          onClick={handleVSCodeClick}
        />
      ) : null}
    </div>
  );
}
