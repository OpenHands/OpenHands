import React from "react";
import { useTranslation } from "react-i18next";

import { CopyToClipboardButton } from "#/components/shared/buttons/copy-to-clipboard-button";
import { I18nKey } from "#/i18n/declaration";

interface WorkspacePathProps {
  path?: string | null;
}

export function WorkspacePath({ path }: WorkspacePathProps) {
  const { t } = useTranslation("openhands");
  const [isCopied, setIsCopied] = React.useState(false);
  const workspacePath = path?.trim();

  React.useEffect(() => {
    if (!isCopied) return undefined;

    const timeout = window.setTimeout(() => setIsCopied(false), 2000);
    return () => window.clearTimeout(timeout);
  }, [isCopied]);

  if (!workspacePath) return null;

  const handleCopy = async () => {
    await navigator.clipboard.writeText(workspacePath);
    setIsCopied(true);
  };

  return (
    <div
      className="flex min-w-0 items-center gap-2 border-b border-[var(--oh-border)] px-3 py-1.5 text-xs"
      data-testid="files-tab-workspace-path"
    >
      <span className="shrink-0 text-[var(--oh-muted)]">
        {t(I18nKey.WORKSPACE$TITLE)}:
      </span>
      <span
        className="min-w-0 flex-1 truncate font-mono"
        data-testid="files-tab-workspace-path-value"
        title={workspacePath}
      >
        {workspacePath}
      </span>
      <CopyToClipboardButton
        isHidden={false}
        isDisabled={isCopied}
        onClick={handleCopy}
        mode={isCopied ? "copied" : "copy"}
      />
    </div>
  );
}
