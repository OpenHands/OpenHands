import React from "react";
import ReactDOM from "react-dom";
import { ExternalLink, Share2 } from "lucide-react";
import { useTranslation } from "react-i18next";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useClickOutsideElement } from "#/hooks/use-click-outside-element";
import { useBreakpoint } from "#/hooks/use-breakpoint";
import { cn } from "#/utils/utils";
import { ContextMenu } from "#/ui/context-menu";
import { ContextMenuListItem } from "../context-menu/context-menu-list-item";
import { Divider } from "#/ui/divider";
import { I18nKey } from "#/i18n/declaration";

import EditIcon from "#/icons/u-edit.svg?react";
import SkillsIcon from "#/icons/skills.svg?react";
import FishingHookIcon from "#/icons/fishing-hook.svg?react";
import ToolsIcon from "#/icons/u-tools.svg?react";
import DownloadIcon from "#/icons/u-download.svg?react";
import CreditCardIcon from "#/icons/u-credit-card.svg?react";
import CloseIcon from "#/icons/u-close.svg?react";
import DeleteIcon from "#/icons/u-delete.svg?react";
import CopyIcon from "#/icons/copy.svg?react";
import { ConversationNameContextMenuIconText } from "./conversation-name-context-menu-icon-text";
import { ArchivedDisabledTooltip } from "../context-menu/archived-disabled-tooltip";
import { useIsArchivedConversation } from "#/hooks/use-is-archived-conversation";

interface ConversationNameContextMenuProps {
  onClose: () => void;
  onRename?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onDelete?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onStop?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onDisplayCost?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onShowAgentTools?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onShowSkills?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onShowHooks?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onTogglePublic?: (nextIsPublic: boolean) => void;
  onCopyShareLink?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onExportTranscript?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onDownloadConversation?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  shareUrl?: string;
  position?: "top" | "bottom";
  /**
   * Element the menu should anchor against. When provided, the menu renders
   * into a portal at the document body using fixed positioning so it cannot be
   * clipped by ancestors with `overflow: hidden` (e.g. the chat panel that
   * sits next to the right-side tabs panel).
   */
  anchorRef?: React.RefObject<HTMLElement | null>;
}

function PublicShareToggle({
  isPublic,
  onToggle,
  ariaLabel,
}: {
  isPublic: boolean;
  onToggle: (nextIsPublic: boolean) => void;
  ariaLabel: string;
}) {
  return (
    <label className="relative inline-flex shrink-0 cursor-pointer items-center">
      <input
        hidden
        type="checkbox"
        data-testid="share-publicly-button"
        checked={isPublic}
        aria-label={ariaLabel}
        onChange={(event) => onToggle(event.target.checked)}
      />
      <span
        aria-hidden
        className={cn(
          "inline-flex h-3.5 w-7 items-center rounded-full px-0.5 py-px transition-colors duration-200 ease-in-out",
          isPublic ? "bg-white" : "bg-base-secondary",
        )}
      >
        <span
          className={cn(
            "block h-2 w-2 shrink-0 rounded-full transition-transform duration-200 ease-in-out",
            isPublic
              ? "translate-x-[calc(1rem-1px)] bg-base-secondary"
              : "translate-x-px bg-tertiary-light",
          )}
        />
      </span>
    </label>
  );
}

export function ConversationNameContextMenu({
  onClose,
  onRename,
  onDelete,
  onStop,
  onDisplayCost,
  onShowAgentTools,
  onShowSkills,
  onShowHooks,
  onTogglePublic,
  onCopyShareLink,
  onExportTranscript,
  onDownloadConversation,
  shareUrl,
  position = "bottom",
  anchorRef,
}: ConversationNameContextMenuProps) {
  const isMobile = useBreakpoint();

  const { t } = useTranslation("openhands");
  const { backend } = useActiveBackend();
  const { data: conversation } = useActiveConversation();
  const isArchivedConversation = useIsArchivedConversation();
  const ref = useClickOutsideElement<HTMLUListElement>(onClose);

  const anchorElement = anchorRef?.current ?? null;
  const [portalStyle, setPortalStyle] = React.useState<React.CSSProperties>();
  React.useLayoutEffect(() => {
    if (!anchorElement) return undefined;

    const updatePosition = () => {
      const rect = anchorElement.getBoundingClientRect();
      if (!rect) return;
      const gap = 8;
      const style: React.CSSProperties = {
        position: "fixed",
        zIndex: 9999,
      };
      if (position === "top") {
        style.bottom = window.innerHeight - rect.top + gap;
      } else {
        style.top = rect.bottom + gap;
      }
      style.left = rect.left;
      setPortalStyle(style);
    };

    updatePosition();
    window.addEventListener("resize", updatePosition);
    window.addEventListener("scroll", updatePosition, true);
    return () => {
      window.removeEventListener("resize", updatePosition);
      window.removeEventListener("scroll", updatePosition, true);
    };
  }, [anchorElement, position]);
  const hasTools = Boolean(onShowAgentTools || onShowSkills || onShowHooks);
  const hasInfo = Boolean(onDisplayCost);
  const hasControl = Boolean(onStop || onDelete);
  const stopLabelKey =
    backend.kind === "cloud"
      ? I18nKey.COMMON$CLOSE_CONVERSATION_STOP_RUNTIME
      : I18nKey.COMMON$STOP_CONVERSATION;
  const shouldShowPublicSharing =
    backend.kind === "cloud" && Boolean(onTogglePublic);
  const isPublic = conversation?.public || false;

  const isPortaled = Boolean(anchorElement);
  const portalClassName = isPortaled
    ? "!static !top-auto !bottom-auto !left-auto !right-auto !mt-0"
    : "";

  const menu = (
    <ContextMenu
      ref={ref}
      testId="conversation-name-context-menu"
      position={position}
      alignment="left"
      className={cn(
        isMobile ? "right-0 translate-x-[34%] left-auto" : "",
        portalClassName,
      )}
    >
      {onRename && (
        <ContextMenuListItem testId="rename-button" onClick={onRename}>
          <ConversationNameContextMenuIconText
            icon={<EditIcon width={16} height={16} />}
            text={t(I18nKey.BUTTON$RENAME)}
          />
        </ContextMenuListItem>
      )}

      {hasTools && <Divider testId="separator-tools" inset="menu" />}

      {onShowSkills && (
        <ArchivedDisabledTooltip isDisabled={isArchivedConversation}>
          <ContextMenuListItem
            testId="show-skills-button"
            onClick={onShowSkills}
            isDisabled={isArchivedConversation}
          >
            <ConversationNameContextMenuIconText
              icon={
                <SkillsIcon
                  width={16}
                  height={16}
                  className="stroke-[1.75]"
                  aria-hidden
                />
              }
              text={t(I18nKey.CONVERSATION$SHOW_SKILLS)}
            />
          </ContextMenuListItem>
        </ArchivedDisabledTooltip>
      )}

      {onShowHooks && (
        <ArchivedDisabledTooltip isDisabled={isArchivedConversation}>
          <ContextMenuListItem
            testId="show-hooks-button"
            onClick={onShowHooks}
            isDisabled={isArchivedConversation}
          >
            <ConversationNameContextMenuIconText
              icon={<FishingHookIcon width={16} height={16} aria-hidden />}
              text={t(I18nKey.CONVERSATION$SHOW_HOOKS)}
            />
          </ContextMenuListItem>
        </ArchivedDisabledTooltip>
      )}

      {onShowAgentTools && (
        <ArchivedDisabledTooltip isDisabled={isArchivedConversation}>
          <ContextMenuListItem
            testId="show-agent-tools-button"
            onClick={onShowAgentTools}
            isDisabled={isArchivedConversation}
          >
            <ConversationNameContextMenuIconText
              icon={<ToolsIcon width={16} height={16} />}
              text={t(I18nKey.BUTTON$SHOW_AGENT_TOOLS_AND_METADATA)}
            />
          </ContextMenuListItem>
        </ArchivedDisabledTooltip>
      )}

      {onExportTranscript && (
        <ContextMenuListItem
          testId="export-transcript-button"
          onClick={onExportTranscript}
        >
          <ConversationNameContextMenuIconText
            icon={<DownloadIcon width={16} height={16} />}
            text={t(I18nKey.BUTTON$EXPORT_TRANSCRIPT)}
          />
        </ContextMenuListItem>
      )}

      {onDownloadConversation && (
        <ContextMenuListItem
          testId="download-trajectory-button"
          onClick={onDownloadConversation}
        >
          <ConversationNameContextMenuIconText
            icon={<DownloadIcon width={16} height={16} />}
            text={t(I18nKey.BUTTON$DOWNLOAD_CONVERSATION_DATA)}
          />
        </ContextMenuListItem>
      )}

      {(hasInfo || hasControl) && (
        <Divider testId="separator-info-control" inset="menu" />
      )}

      {onDisplayCost && (
        <ContextMenuListItem
          testId="display-cost-button"
          onClick={onDisplayCost}
        >
          <ConversationNameContextMenuIconText
            icon={<CreditCardIcon width={16} height={16} />}
            text={t(I18nKey.BUTTON$DISPLAY_COST)}
          />
        </ContextMenuListItem>
      )}

      {shouldShowPublicSharing && onTogglePublic && (
        <li className="flex w-full items-center gap-2 rounded px-2 py-2 hover:bg-[var(--oh-interactive-hover)]">
          <span
            className="flex shrink-0 items-center text-[var(--oh-muted)]"
            aria-hidden
          >
            <Share2 size={16} />
          </span>
          <span className="min-w-0 flex-1 truncate text-sm">
            {t(I18nKey.CONVERSATION$SHARE_PUBLICLY)}
          </span>
          <div className="flex shrink-0 items-center">
            {isPublic && shareUrl && onCopyShareLink && (
              <div className="mr-2 flex items-center gap-0.5">
                <button
                  type="button"
                  data-testid="copy-share-link-button"
                  onClick={onCopyShareLink}
                  className="rounded p-0.5 text-[var(--oh-muted)] hover:bg-[var(--oh-interactive-selected)] hover:text-[var(--oh-foreground)] cursor-pointer [&_svg]:text-current"
                  title={t(I18nKey.BUTTON$COPY_TO_CLIPBOARD)}
                >
                  <CopyIcon width={14} height={14} />
                </button>
                <a
                  data-testid="open-share-link-button"
                  href={shareUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  onClick={(event) => event.stopPropagation()}
                  className="rounded p-0.5 text-[var(--oh-muted)] no-underline visited:text-[var(--oh-muted)] hover:bg-[var(--oh-interactive-selected)] hover:text-[var(--oh-foreground)] cursor-pointer [&_svg]:text-current"
                  title={t(I18nKey.BUTTON$OPEN_IN_NEW_TAB)}
                >
                  <ExternalLink size={14} aria-hidden />
                </a>
              </div>
            )}
            <PublicShareToggle
              isPublic={isPublic}
              onToggle={onTogglePublic}
              ariaLabel={t(I18nKey.CONVERSATION$SHARE_PUBLICLY)}
            />
          </div>
        </li>
      )}

      {onStop && (
        <ArchivedDisabledTooltip isDisabled={isArchivedConversation}>
          <ContextMenuListItem
            testId="stop-button"
            onClick={onStop}
            isDisabled={isArchivedConversation}
          >
            <ConversationNameContextMenuIconText
              icon={<CloseIcon width={16} height={16} />}
              text={t(stopLabelKey)}
            />
          </ContextMenuListItem>
        </ArchivedDisabledTooltip>
      )}

      {onDelete && (
        <ContextMenuListItem testId="delete-button" onClick={onDelete}>
          <ConversationNameContextMenuIconText
            icon={<DeleteIcon width={16} height={16} />}
            text={t(I18nKey.COMMON$DELETE_CONVERSATION)}
          />
        </ContextMenuListItem>
      )}
    </ContextMenu>
  );

  if (isPortaled) {
    if (typeof document === "undefined" || !portalStyle) {
      return null;
    }
    return ReactDOM.createPortal(
      <div style={portalStyle}>{menu}</div>,
      document.body,
    );
  }

  return menu;
}
