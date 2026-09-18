import { useTranslation } from "react-i18next";
import { EyeOff } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import {
  dropdownMenuRowClassName,
  dropdownMenuRowIconClassName,
} from "#/utils/dropdown-classes";
import { ToggleSwitchVisual } from "#/ui/toggle-switch";
import { EnumFilterDropdown } from "#/components/shared/filters/enum-filter-dropdown";
import { useConversationPanelPreferencesStore } from "#/stores/conversation-panel-preferences-store";
import {
  DEFAULT_OLDER_CONVERSATION_CUTOFF,
  isOlderConversationCutoff,
  OLDER_CONVERSATION_CUTOFFS,
} from "./conversation-panel-list-helpers";

const CUTOFF_LABEL_KEYS = {
  "1h": I18nKey.CONVERSATION_PANEL$OLDER_OVER_ONE_HOUR,
  "1d": I18nKey.CONVERSATION_PANEL$OLDER_OVER_ONE_DAY,
  "7d": I18nKey.CONVERSATION_PANEL$OLDER_OVER_ONE_WEEK,
  "30d": I18nKey.CONVERSATION_PANEL$OLDER_OVER_THIRTY_DAYS,
} as const;

/**
 * Hide-older control: a toggle plus an interval select. The persisted
 * preference is still `showOlderConversations` (true = show everything);
 * the toggle is inverted so on means hide.
 */
export function HideOlderConversationsRow() {
  const { t } = useTranslation("openhands");
  const showOlderConversations = useConversationPanelPreferencesStore(
    (state) => state.showOlderConversations,
  );
  const olderConversationCutoff = useConversationPanelPreferencesStore(
    (state) => state.olderConversationCutoff,
  );
  const toggleShowOlderConversations = useConversationPanelPreferencesStore(
    (state) => state.toggleShowOlderConversations,
  );
  const setOlderConversationCutoff = useConversationPanelPreferencesStore(
    (state) => state.setOlderConversationCutoff,
  );

  const hideOlder = !showOlderConversations;
  const cutoff = isOlderConversationCutoff(olderConversationCutoff)
    ? olderConversationCutoff
    : DEFAULT_OLDER_CONVERSATION_CUTOFF;

  return (
    <div
      role="menuitemcheckbox"
      aria-checked={hideOlder}
      data-testid="toggle-older-conversations"
      tabIndex={0}
      onClick={toggleShowOlderConversations}
      onKeyDown={(event) => {
        if (event.key === " " || event.key === "Enter") {
          event.preventDefault();
          toggleShowOlderConversations();
        }
      }}
      className={cn(
        "group",
        dropdownMenuRowClassName,
        "text-[var(--oh-foreground)]",
      )}
    >
      <EyeOff
        className={cn("h-3.5 w-3.5", dropdownMenuRowIconClassName)}
        aria-hidden
      />
      <span className="min-w-0 shrink truncate">
        {t(I18nKey.CONVERSATION_PANEL$HIDE_CONVERSATIONS)}
      </span>
      <div
        className="shrink-0"
        onClick={(event) => event.stopPropagation()}
        onKeyDown={(event) => event.stopPropagation()}
      >
        <EnumFilterDropdown
          testId="older-conversation-cutoff"
          value={cutoff}
          onChange={setOlderConversationCutoff}
          options={OLDER_CONVERSATION_CUTOFFS}
          labelKeyByValue={CUTOFF_LABEL_KEYS}
          ariaLabel={t(I18nKey.CONVERSATION_PANEL$OLDER_CUTOFF_LABEL)}
          emphasizeNonDefault={false}
          triggerClassName={cn(
            "h-7 gap-1 rounded py-0 pl-2 pr-1.5 font-normal",
            "text-[var(--oh-foreground)] [&_svg]:h-3 [&_svg]:w-3 [&_svg]:text-[var(--oh-muted)]",
          )}
        />
      </div>
      <ToggleSwitchVisual enabled={hideOlder} size="sm" className="ml-auto" />
    </div>
  );
}
