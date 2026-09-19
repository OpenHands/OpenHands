import React, { useEffect, useRef, useState } from "react";
import { GitCommitHorizontal } from "lucide-react";
import { LuFileDiff } from "react-icons/lu";
import { useTranslation } from "react-i18next";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useConversationOverviewGitDiffStats } from "#/hooks/use-conversation-overview-git-diff-stats";
import { useSelectConversationTab } from "#/hooks/use-select-conversation-tab";
import { I18nKey } from "#/i18n/declaration";
import {
  OH_STATUS_ERROR_COLOR,
  OH_STATUS_SUCCESS_COLOR,
} from "#/constants/status-colors";
import { cn } from "#/utils/utils";
import { Provider } from "#/types/settings";
import { useConversationOverviewDrawerOptional } from "./conversation-overview-drawer-context";
import { ConversationGitActionsMenu } from "./conversation-git-actions-menu";

const CHANGES_BUTTON_CLASSNAME = cn(
  "flex min-w-0 flex-1 items-center gap-2 rounded-md px-2 py-1.5",
  "cursor-pointer border-0 bg-transparent text-left",
);

const ROW_ICON_CLASSNAME = "size-4 shrink-0 text-[var(--oh-muted)]";

const GIT_ACTION_HOVER_TARGET_CLASSNAME =
  "conversation-overview-diffs-git-action";

const GIT_ACTION_BUTTON_CLASSNAME = cn(
  GIT_ACTION_HOVER_TARGET_CLASSNAME,
  "absolute inset-0 inline-flex items-center justify-center",
  "rounded-md text-[var(--oh-muted)] transition-opacity",
  "hover:bg-white/10 hover:text-[var(--oh-foreground)]",
);

/** Full-row hover, cleared while the git commit control itself is hovered. */
const DIFFS_ROW_CLASSNAME = cn(
  "flex items-center rounded-md transition-colors",
  "hover:bg-white/5",
  `has-[.${GIT_ACTION_HOVER_TARGET_CLASSNAME}:hover]:bg-transparent`,
);

const GIT_ACTION_OVERLAY_CLASSNAME = cn(
  "pointer-events-none invisible opacity-0",
  "group-hover/diffstats:pointer-events-auto group-hover/diffstats:visible group-hover/diffstats:opacity-100",
  "group-focus-within/diffstats:pointer-events-auto group-focus-within/diffstats:visible group-focus-within/diffstats:opacity-100",
);

export function ConversationOverviewDiffsRow() {
  const { t } = useTranslation("openhands");
  const { navigateToChanges } = useSelectConversationTab();
  const overviewDrawer = useConversationOverviewDrawerOptional();
  const { data: conversation } = useActiveConversation();
  const { additions, deletions, isLoading } =
    useConversationOverviewGitDiffStats();
  const gitProvider =
    (conversation?.git_provider as Provider | undefined) ?? "github";
  const [isGitMenuOpen, setIsGitMenuOpen] = useState(false);
  const gitActionButtonRef = useRef<HTMLButtonElement>(null);

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

  const handleOpenChanges = () => {
    overviewDrawer?.closeDrawer();
    navigateToChanges();
  };

  return (
    <li className={DIFFS_ROW_CLASSNAME}>
      <button
        type="button"
        data-testid="conversation-overview-diffs"
        aria-label={t(I18nKey.CONVERSATION$OVERVIEW_OPEN_CHANGES)}
        onClick={handleOpenChanges}
        className={CHANGES_BUTTON_CLASSNAME}
      >
        <LuFileDiff className={ROW_ICON_CLASSNAME} aria-hidden />
        <span className="min-w-0 flex-1 truncate text-sm text-[var(--oh-foreground)]">
          {t(I18nKey.COMMON$CHANGES)}
        </span>
      </button>

      <div className="group/diffstats relative mr-2 flex h-6 min-w-6 shrink-0 items-center justify-center">
        {isLoading ? (
          <span className="text-sm tabular-nums text-[var(--oh-muted)]">…</span>
        ) : (
          <>
            <span
              className={cn(
                "flex items-center gap-1.5 text-sm tabular-nums transition-opacity",
                isGitMenuOpen
                  ? "opacity-0"
                  : "group-hover/diffstats:opacity-0 group-focus-within/diffstats:opacity-0",
              )}
            >
              <span
                data-testid="conversation-overview-diffs-additions"
                style={{ color: OH_STATUS_SUCCESS_COLOR }}
              >
                {t(I18nKey.CONVERSATION$OVERVIEW_DIFF_ADDITIONS, {
                  count: additions.toLocaleString(),
                })}
              </span>
              <span
                data-testid="conversation-overview-diffs-deletions"
                style={{ color: OH_STATUS_ERROR_COLOR }}
              >
                {t(I18nKey.CONVERSATION$OVERVIEW_DIFF_DELETIONS, {
                  count: deletions.toLocaleString(),
                })}
              </span>
            </span>
            <button
              ref={gitActionButtonRef}
              type="button"
              data-testid="conversation-overview-diffs-git-action"
              aria-label={t(I18nKey.CONVERSATION$OVERVIEW_DIFF_GIT_ACTIONS)}
              aria-expanded={isGitMenuOpen}
              aria-haspopup="menu"
              onClick={(event) => {
                event.stopPropagation();
                setIsGitMenuOpen((open) => !open);
              }}
              className={cn(
                GIT_ACTION_BUTTON_CLASSNAME,
                !isGitMenuOpen && GIT_ACTION_OVERLAY_CLASSNAME,
                isGitMenuOpen &&
                  "pointer-events-auto visible opacity-100 text-[var(--oh-foreground)]",
              )}
            >
              <GitCommitHorizontal className="size-4" aria-hidden />
            </button>
            {isGitMenuOpen ? (
              <ConversationGitActionsMenu
                anchorRef={gitActionButtonRef}
                gitProvider={gitProvider}
                testIdPrefix="conversation-overview-diffs-git"
                onClose={() => setIsGitMenuOpen(false)}
              />
            ) : null}
          </>
        )}
      </div>
    </li>
  );
}
