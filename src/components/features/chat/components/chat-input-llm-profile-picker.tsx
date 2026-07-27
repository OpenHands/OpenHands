import React from "react";
import { useTranslation } from "react-i18next";
import { useChatInputLlmProfileState } from "#/hooks/use-chat-input-llm-profile-state";
import { ComboboxCaretInline } from "#/ui/combobox-caret";
import SettingsGearIcon from "#/icons/settings-gear.svg?react";
import CheckIcon from "#/icons/checkmark.svg?react";
import { useClickOutsideElement } from "#/hooks/use-click-outside-element";
import { NavigationLink } from "#/components/shared/navigation-link";
import { ContextMenu } from "#/ui/context-menu";
import { ContextMenuListItem } from "#/components/features/context-menu/context-menu-list-item";
import { Divider } from "#/ui/divider";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { chatInputPillButtonClassName } from "#/utils/form-control-classes";

const PROFILE_LABEL_MAX_CHARS = 18;

function truncateLabel(label: string): string {
  return label.length <= PROFILE_LABEL_MAX_CHARS
    ? label
    : `${label.slice(0, PROFILE_LABEL_MAX_CHARS)}…`;
}

interface ChatInputLlmProfileMenuContentProps {
  onClose: () => void;
  dividerInset?: "menu";
  settingsLinkClassName?: string;
  settingsIconClassName?: string;
}

/**
 * The in-conversation OpenHands LLM-profile switcher list. Selecting a profile
 * live-swaps the running conversation's LLM via `/switch_profile` (the ACP
 * analog is {@link ChatInputModelMenuContent}). Shared by the inline pill and
 * the chat-input overflow submenu.
 */
export function ChatInputLlmProfileMenuContent({
  onClose,
  dividerInset,
  settingsLinkClassName,
  settingsIconClassName,
}: ChatInputLlmProfileMenuContentProps) {
  const { t } = useTranslation("openhands");
  const { profiles, currentProfileName, selectProfile } =
    useChatInputLlmProfileState();

  const handleSelect = (profileName: string) => {
    selectProfile(profileName);
    onClose();
  };

  return (
    <>
      {profiles.length > 0 && (
        <>
          {/* role="presentation" keeps this a valid <li> child of the
              ContextMenu <ul> without exposing the label as a menu item. */}
          <li role="presentation" className="px-2 pt-1 pb-0.5">
            <Typography.Text className="text-[11px] font-medium text-[var(--oh-text-dim)] uppercase tracking-wide leading-4">
              {t(I18nKey.SETTINGS$AVAILABLE_PROFILES)}
            </Typography.Text>
          </li>
          {profiles.map((profile) => {
            const isCurrent = profile.name === currentProfileName;
            return (
              <ContextMenuListItem
                key={profile.name}
                testId={`chat-input-llm-profile-option-${profile.name}`}
                onClick={(event) => {
                  event.preventDefault();
                  event.stopPropagation();
                  if (isCurrent) {
                    onClose();
                    return;
                  }
                  handleSelect(profile.name);
                }}
                className={cn(
                  "flex flex-col items-stretch gap-0.5",
                  isCurrent && "bg-[var(--oh-interactive-hover)]",
                )}
              >
                <span className="flex items-center gap-2">
                  <span
                    className="flex-1 truncate text-sm leading-5"
                    title={profile.model ?? profile.name}
                  >
                    {profile.name}
                  </span>
                  {isCurrent && (
                    <CheckIcon
                      width={14}
                      height={14}
                      className="shrink-0"
                      aria-hidden
                    />
                  )}
                </span>
                {profile.model && (
                  <span className="block truncate text-xs leading-4 text-[var(--oh-muted)]">
                    {profile.model}
                  </span>
                )}
              </ContextMenuListItem>
            );
          })}
        </>
      )}
      {profiles.length > 0 && <Divider inset={dividerInset} />}
      <li className="text-sm">
        <NavigationLink
          to="/settings/llm"
          onClick={onClose}
          className={cn(
            "flex h-[30px] items-center gap-2 rounded p-2 leading-5 text-[var(--oh-foreground)] hover:bg-[var(--oh-interactive-hover)] transition-colors",
            settingsLinkClassName,
          )}
        >
          <SettingsGearIcon
            width={16}
            height={16}
            className={cn("shrink-0", settingsIconClassName)}
            aria-hidden
          />
          <span>{t(I18nKey.SETTINGS$LLM_PROFILES)}</span>
        </NavigationLink>
      </li>
    </>
  );
}

export function ChatInputLlmProfilePicker() {
  const { t } = useTranslation("openhands");
  const { profiles, currentProfileName, isLoading, isSwitching } =
    useChatInputLlmProfileState();
  const [isPopoverOpen, setIsPopoverOpen] = React.useState(false);
  const triggerRef = React.useRef<HTMLButtonElement>(null);
  const popoverRef = useClickOutsideElement<HTMLUListElement>(
    () => setIsPopoverOpen(false),
    triggerRef,
  );

  // No LLM profiles yet (or the agent-server lacks the surface): stay out of
  // the way, exactly like the ACP/AgentProfile pickers.
  if (isLoading || profiles.length === 0) {
    return null;
  }

  const label = currentProfileName ?? t(I18nKey.LLM$SELECT_MODEL_PLACEHOLDER);

  return (
    <div className="relative min-w-0">
      <button
        ref={triggerRef}
        type="button"
        className={cn(chatInputPillButtonClassName, "max-w-[200px]")}
        title={currentProfileName ?? undefined}
        data-testid="chat-input-llm-profile"
        aria-expanded={isPopoverOpen}
        aria-haspopup="dialog"
        // Disabled mid-switch so re-opening can't fire a second /switch_profile.
        disabled={isSwitching}
        aria-busy={isSwitching}
        onClick={(event) => {
          event.preventDefault();
          event.stopPropagation();
          setIsPopoverOpen((open) => !open);
        }}
      >
        <span className="truncate">{truncateLabel(label)}</span>
        <ComboboxCaretInline isOpen={isPopoverOpen} />
      </button>

      {isPopoverOpen && (
        <ContextMenu
          ref={popoverRef}
          testId="chat-input-llm-profile-popover"
          position="top"
          alignment="left"
          spacing="none"
          className="z-[60] mb-2 min-w-[200px] max-w-[320px] max-h-[60vh] overflow-y-auto"
        >
          <ChatInputLlmProfileMenuContent
            onClose={() => setIsPopoverOpen(false)}
          />
        </ContextMenu>
      )}
    </div>
  );
}
