import React from "react";
import { useTranslation } from "react-i18next";
import { useChatInputProfileState } from "#/hooks/use-chat-input-profile-state";
import { useClickOutsideElement } from "#/hooks/use-click-outside-element";
import RobotIcon from "#/icons/u-robot.svg?react";
import { ContextMenu } from "#/ui/context-menu";
import { ComboboxCaretInline } from "#/ui/combobox-caret";
import { chatInputPillButtonClassName } from "#/utils/form-control-classes";
import SettingsGearIcon from "#/icons/settings-gear.svg?react";
import CheckIcon from "#/icons/checkmark.svg?react";
import { NavigationLink } from "#/components/shared/navigation-link";
import { ContextMenuListItem } from "#/components/features/context-menu/context-menu-list-item";
import { Divider } from "#/ui/divider";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { getLockedCloudHost } from "#/api/agent-server-config";

interface ChatInputProfileMenuContentProps {
  onClose: () => void;
  dividerInset?: "menu";
  settingsLinkClassName?: string;
  settingsIconClassName?: string;
}

/**
 * The agent-profile list rendered inside the "+" tools menu's "Switch agent
 * profile" submenu, offered only while starting a new conversation (OSS-5735 —
 * the profile is locked once a conversation starts). Selecting a profile
 * activates it (home) or recreates the blank conversation with it (see
 * `useChatInputProfileState`). Also the menu of the
 * {@link ChatInputAgentProfilePicker} pill.
 */
export function ChatInputProfileMenuContent({
  onClose,
  dividerInset,
  settingsLinkClassName,
  settingsIconClassName,
}: ChatInputProfileMenuContentProps) {
  const { t } = useTranslation("openhands");
  const {
    profiles,
    currentProfileId,
    isInConversation,
    isSwitching,
    selectProfile,
  } = useChatInputProfileState();
  // Locked-to-Cloud blocks the Canvas agent profiles page this links to
  // (OHE-3457).
  const showSettingsLink = getLockedCloudHost() === null;

  const handleSelect = (profile: (typeof profiles)[number]) => {
    selectProfile(profile);
    onClose();
  };

  return (
    <>
      {profiles.length > 0 && (
        <>
          {/* role="presentation" keeps this a valid <li> child of the
              ContextMenu <ul> without exposing the label as a menu item. */}
          <li role="presentation" className="px-2 pt-1 pb-0.5">
            <Typography.Text className="text-[11px] font-medium text-text-dim uppercase tracking-wide leading-4">
              {t(I18nKey.SETTINGS$AVAILABLE_PROFILES)}
            </Typography.Text>
          </li>
          {profiles.map((profile) => {
            const isCurrent =
              profile.id != null && profile.id === currentProfileId;
            return (
              <ContextMenuListItem
                key={profile.id ?? profile.name}
                testId={`chat-input-agent-profile-option-${profile.name}`}
                isDisabled={isSwitching}
                onClick={(event) => {
                  event.preventDefault();
                  event.stopPropagation();
                  if (isCurrent) {
                    onClose();
                    return;
                  }
                  handleSelect(profile);
                }}
                className={cn(
                  "flex items-center gap-2",
                  isCurrent && "bg-interactive-hover",
                )}
              >
                <span
                  className="flex-1 truncate text-sm leading-5"
                  title={profile.name}
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
              </ContextMenuListItem>
            );
          })}
          {isInConversation && (
            <li role="presentation" className="px-2 pt-0.5 pb-1">
              <Typography.Text className="text-[11px] text-text-dim leading-4">
                {t(I18nKey.CHAT$START_NEW_WITH_PROFILE_HINT)}
              </Typography.Text>
            </li>
          )}
        </>
      )}
      {showSettingsLink && profiles.length > 0 && (
        <Divider inset={dividerInset} />
      )}
      {showSettingsLink && (
        <li className="text-sm">
          <NavigationLink
            to="/agents"
            onClick={onClose}
            className={cn(
              "flex h-7.5 items-center gap-2 rounded p-2 leading-5 text-foreground hover:bg-interactive-hover transition-colors",
              settingsLinkClassName,
            )}
          >
            <SettingsGearIcon
              width={16}
              height={16}
              className={cn("shrink-0", settingsIconClassName)}
              aria-hidden
            />
            <span>{t(I18nKey.CHAT$MANAGE_AGENT_PROFILES)}</span>
          </NavigationLink>
        </li>
      )}
    </>
  );
}

/** Names the agent profile a new conversation will run, and switches it. */
export function ChatInputAgentProfilePicker() {
  const { t } = useTranslation("openhands");
  const { profiles, currentProfileName, isLoading, isSwitching } =
    useChatInputProfileState();
  const [isOpen, setIsOpen] = React.useState(false);
  const triggerRef = React.useRef<HTMLButtonElement>(null);
  const popoverRef = useClickOutsideElement<HTMLUListElement>(
    () => setIsOpen(false),
    triggerRef,
  );

  // No profiles yet: stay out of the way, like the LLM picker. With profiles
  // but nothing selected, the pill still renders so the user can pick one.
  if (isLoading || profiles.length === 0) return null;

  const label = currentProfileName ?? t(I18nKey.CHAT$AGENT_PROFILE_PLACEHOLDER);

  return (
    <div className="relative min-w-0">
      <button
        ref={triggerRef}
        type="button"
        className={cn(chatInputPillButtonClassName, "max-w-40")}
        title={t(I18nKey.CHAT$SWITCH_AGENT_PROFILE)}
        aria-label={`${t(I18nKey.CHAT$SWITCH_AGENT_PROFILE)}: ${label}`}
        data-testid="chat-input-agent-profile"
        aria-expanded={isOpen}
        aria-haspopup="dialog"
        disabled={isSwitching}
        aria-busy={isSwitching}
        onClick={(event) => {
          event.preventDefault();
          event.stopPropagation();
          setIsOpen((open) => !open);
        }}
      >
        <RobotIcon width={14} height={14} className="shrink-0" aria-hidden />
        <span className="truncate">{label}</span>
        <ComboboxCaretInline isOpen={isOpen} />
      </button>
      {isOpen && (
        <ContextMenu
          ref={popoverRef}
          testId="chat-input-agent-profile-popover"
          position="top"
          alignment="left"
          spacing="none"
          className="z-[60] mb-2 min-w-50 max-w-80 max-h-[60vh] overflow-y-auto"
        >
          <ChatInputProfileMenuContent onClose={() => setIsOpen(false)} />
        </ContextMenu>
      )}
    </div>
  );
}
