import { useTranslation } from "react-i18next";
import { useClickOutsideElement } from "#/hooks/use-click-outside-element";
import { ContextMenu } from "#/ui/context-menu";
import { I18nKey } from "#/i18n/declaration";
import StopCircleIcon from "#/icons/stop-circle.svg?react";
import PlayCircleIcon from "#/icons/play-circle.svg?react";
import { ServerStatusContextMenuIconText } from "./server-status-context-menu-icon-text";
import { ServerStatus } from "./server-status";
import { Divider } from "#/ui/divider";
import { cn } from "#/utils/utils";
import { V1SandboxStatus } from "#/api/sandbox-service/sandbox-service.types";

interface ServerStatusContextMenuProps {
  onClose: () => void;
  onStopServer?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onStartServer?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  sandboxStatus: V1SandboxStatus | null;
  position?: "top" | "bottom";
  className?: string;
  isPausing?: boolean;
}

export function ServerStatusContextMenu({
  onClose,
  onStopServer,
  onStartServer,
  sandboxStatus,
  position = "top",
  className = "",
  isPausing = false,
}: ServerStatusContextMenuProps) {
  const { t } = useTranslation();
  const ref = useClickOutsideElement<HTMLUListElement>(onClose);

  const shouldActionShown =
    sandboxStatus === "RUNNING" || sandboxStatus === "MISSING";

  return (
    <ContextMenu
      ref={ref}
      testId="server-status-context-menu"
      position={position}
      alignment="left"
      size="default"
      className={cn("left-2 w-fit min-w-42", className)}
    >
      <ServerStatus
        sandboxStatus={sandboxStatus}
        isPausing={isPausing}
        className="py-1"
      />

      {shouldActionShown && (
        <>
          <Divider />

          {sandboxStatus === "RUNNING" && onStopServer && (
            <ServerStatusContextMenuIconText
              icon={<StopCircleIcon width={18} height={18} />}
              text={t(I18nKey.COMMON$STOP_RUNTIME)}
              onClick={onStopServer}
              testId="stop-server-button"
            />
          )}

          {sandboxStatus === "MISSING" && onStartServer && (
            <ServerStatusContextMenuIconText
              icon={<PlayCircleIcon width={18} height={18} />}
              text={t(I18nKey.COMMON$START_RUNTIME)}
              onClick={onStartServer}
              testId="start-server-button"
            />
          )}
        </>
      )}
    </ContextMenu>
  );
}
