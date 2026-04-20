import { FaExternalLinkAlt, FaDesktop } from "react-icons/fa";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { useAgentState } from "#/hooks/use-agent-state";
import { useUnifiedVSCodeUrl } from "#/hooks/query/use-unified-vscode-url";
import { useSSHConnection } from "#/hooks/query/use-ssh-connection";
import { RUNTIME_STARTING_STATES } from "#/types/agent-state";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

export function VSCodeTooltipContent() {
  const { curAgentState } = useAgentState();
  const { t } = useTranslation();
  const { data: vscodeData, refetch: refetchVSCode } = useUnifiedVSCodeUrl();
  const { data: sshData, refetch: refetchSSH } = useSSHConnection();
  const isRuntimeStarting = RUNTIME_STARTING_STATES.includes(curAgentState);

  const handleVSCodeClick = async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();

    let vscodeUrl = vscodeData?.url;

    if (!vscodeUrl) {
      const result = await refetchVSCode();
      vscodeUrl = result.data?.url ?? null;
    }

    if (vscodeUrl) {
      window.open(vscodeUrl, "_blank", "noopener,noreferrer");
    }
  };

  const handleLocalVSCodeClick = async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();

    // Refetch to get latest status
    const result = await refetchSSH();
    const currentData = result.data ?? sshData;

    // Check for errors and show toast if SSH is not available
    if (currentData?.error) {
      displayErrorToast(currentData.error);
      return;
    }

    if (currentData?.vscodeRemoteUrl) {
      window.location.href = currentData.vscodeRemoteUrl;
    } else {
      displayErrorToast(t(I18nKey.SSH$URL_NOT_AVAILABLE));
    }
  };

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2">
        <span>{t(I18nKey.COMMON$CODE)}</span>
        {!isRuntimeStarting ? (
          <FaExternalLinkAlt
            className="w-3 h-3 text-inherit cursor-pointer"
            onClick={handleVSCodeClick}
            title={t(I18nKey.VSCODE$OPEN_IN_NEW_TAB)}
          />
        ) : null}
      </div>
      {!isRuntimeStarting ? (
        <div
          className="flex items-center gap-2 cursor-pointer text-xs opacity-80 hover:opacity-100"
          onClick={handleLocalVSCodeClick}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              handleLocalVSCodeClick(e as unknown as React.MouseEvent);
            }
          }}
        >
          <FaDesktop className="w-3 h-3 text-inherit" />
          <span>{t(I18nKey.VSCODE$OPEN_IN_LOCAL_VSCODE)}</span>
        </div>
      ) : null}
    </div>
  );
}
