import { useTranslation } from "react-i18next";
import { useChatInputProfileState } from "#/hooks/use-chat-input-profile-state";
import SettingsGearIcon from "#/icons/settings-gear.svg?react";
import CheckIcon from "#/icons/checkmark.svg?react";
import { NavigationLink } from "#/components/shared/navigation-link";
import { ContextMenuListItem } from "#/components/features/context-menu/context-menu-list-item";
import { Divider } from "#/ui/divider";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

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
 * `useChatInputProfileState`). The chat-input pill itself is always an LLM
 * selector; the former `ChatInputProfilePicker` pill is gone.
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
            <Typography.Text className="text-[11px] font-medium text-[var(--oh-text-dim)] uppercase tracking-wide leading-4">
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
                  isCurrent && "bg-[var(--oh-interactive-hover)]",
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
              <Typography.Text className="text-[11px] text-[var(--oh-text-dim)] leading-4">
                {t(I18nKey.CHAT$START_NEW_WITH_PROFILE_HINT)}
              </Typography.Text>
            </li>
          )}
        </>
      )}
      {profiles.length > 0 && <Divider inset={dividerInset} />}
      <li className="text-sm">
        <NavigationLink
          to="/settings/agents"
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
          <span>{t(I18nKey.CHAT$MANAGE_AGENT_PROFILES)}</span>
        </NavigationLink>
      </li>
    </>
  );
}
