import React from "react";
import { useTranslation } from "react-i18next";
import { useChatInputProfileState } from "#/hooks/use-chat-input-profile-state";
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

interface ChatInputProfileMenuContentProps {
  onClose: () => void;
  dividerInset?: "menu";
  settingsLinkClassName?: string;
  settingsIconClassName?: string;
}

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

export function ChatInputProfilePicker() {
  const { t } = useTranslation("openhands");
  const { profiles, currentProfileName, isLoading, isSwitching } =
    useChatInputProfileState();
  const [isPopoverOpen, setIsPopoverOpen] = React.useState(false);
  const triggerRef = React.useRef<HTMLButtonElement>(null);
  const popoverRef = useClickOutsideElement<HTMLUListElement>(
    () => setIsPopoverOpen(false),
    triggerRef,
  );

  // Nothing to launch from yet (empty store before the backend seeds, or an
  // agent-server without the /api/agent-profiles surface): stay out of the way.
  if (isLoading || profiles.length === 0) {
    return null;
  }

  const label = currentProfileName ?? t(I18nKey.CHAT$AGENT_PROFILE_PLACEHOLDER);

  return (
    <div className="relative min-w-0">
      <button
        ref={triggerRef}
        type="button"
        className={cn(chatInputPillButtonClassName, "max-w-[200px]")}
        title={currentProfileName ?? undefined}
        data-testid="chat-input-agent-profile"
        aria-expanded={isPopoverOpen}
        aria-haspopup="dialog"
        // Disabled while a profile switch (new-conversation create / activate) is
        // in flight, so re-opening the menu can't fire a second create. The menu
        // closes on select, so this is the only double-submit path.
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
          testId="chat-input-agent-profile-popover"
          position="top"
          alignment="left"
          spacing="none"
          className="z-[60] mb-2 min-w-[200px] max-w-[320px] max-h-[60vh] overflow-y-auto"
        >
          <ChatInputProfileMenuContent
            onClose={() => setIsPopoverOpen(false)}
          />
        </ContextMenu>
      )}
    </div>
  );
}
