import React, { useEffect, useRef, useState } from "react";
import { GitCommitHorizontal } from "lucide-react";
import { useTranslation } from "react-i18next";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useIsArchivedConversation } from "#/hooks/use-is-archived-conversation";
import { I18nKey } from "#/i18n/declaration";
import { Provider } from "#/types/settings";
import { cn } from "#/utils/utils";
import {
  formControlBorderClassName,
  formControlMutedHoverClassName,
  formControlTransitionClassName,
} from "#/utils/form-control-classes";
import { ChatActionTooltip } from "../chat/chat-action-tooltip";
import { ConversationGitActionsMenu } from "./conversation-git-actions-menu";

/** Same 28px height as overview / drawer header icon buttons. */
const GIT_ACTIONS_BUTTON_CLASSNAME = cn(
  "inline-flex h-7 min-h-7 w-fit shrink-0 cursor-pointer items-center justify-center gap-1.5 px-2.5",
  "rounded-md text-xs font-normal leading-none",
  formControlBorderClassName,
  formControlTransitionClassName,
  "text-[var(--oh-muted)]",
  formControlMutedHoverClassName,
  "disabled:cursor-not-allowed disabled:opacity-30",
);

interface ConversationGitActionsToggleProps {
  className?: string;
}

/**
 * Header control left of overview: opens the shared git-actions dropdown
 * (commit / pull / push / create PR / new branch) as chat prompts.
 */
export function ConversationGitActionsToggle({
  className,
}: ConversationGitActionsToggleProps) {
  const { t } = useTranslation("openhands");
  const isArchivedConversation = useIsArchivedConversation();
  const { data: conversation } = useActiveConversation();
  const [isGitMenuOpen, setIsGitMenuOpen] = useState(false);
  const buttonRef = useRef<HTMLButtonElement>(null);

  const gitProvider =
    (conversation?.git_provider as Provider | undefined) ?? "github";

  useEffect(() => {
    if (!isGitMenuOpen) {
      return undefined;
    }

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setIsGitMenuOpen(false);
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [isGitMenuOpen]);

  const label = t(I18nKey.CONVERSATION$OVERVIEW_DIFF_GIT_ACTIONS);

  const button = (
    <button
      ref={buttonRef}
      type="button"
      onClick={() => {
        if (isArchivedConversation) {
          return;
        }
        setIsGitMenuOpen((open) => !open);
      }}
      disabled={isArchivedConversation}
      className={cn(
        GIT_ACTIONS_BUTTON_CLASSNAME,
        isGitMenuOpen && "bg-white/10 text-[var(--oh-foreground)]",
        isArchivedConversation &&
          "cursor-not-allowed opacity-50 hover:bg-transparent hover:text-[var(--oh-muted)]",
        className,
      )}
      aria-expanded={isGitMenuOpen}
      aria-haspopup="menu"
      aria-disabled={isArchivedConversation}
      data-testid="conversation-git-actions-toggle"
    >
      <GitCommitHorizontal className="size-4 shrink-0" size={16} aria-hidden />
      <span className="whitespace-nowrap">{label}</span>
    </button>
  );

  return (
    <div className="relative inline-flex items-center self-center">
      {/* Label is visible, so a tooltip only adds value for the disabled reason. */}
      {isArchivedConversation ? (
        <ChatActionTooltip
          tooltip={t(I18nKey.CONVERSATION$UNAVAILABLE_FOR_ARCHIVES)}
          ariaLabel={t(I18nKey.CONVERSATION$UNAVAILABLE_FOR_ARCHIVES)}
        >
          {button}
        </ChatActionTooltip>
      ) : (
        button
      )}
      {isGitMenuOpen ? (
        <ConversationGitActionsMenu
          anchorRef={buttonRef}
          gitProvider={gitProvider}
          onClose={() => setIsGitMenuOpen(false)}
        />
      ) : null}
    </div>
  );
}
