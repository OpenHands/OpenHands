import { useTranslation } from "react-i18next";
import {
  useChatInputModelState,
  type ChatInputModelState,
} from "#/hooks/use-chat-input-model-state";
import { useSwitchAcpModel } from "#/hooks/mutation/use-switch-acp-model";
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
import React from "react";

const MODEL_LABEL_MAX_CHARS = 10;
// ACP surfaces show the provider's human label (e.g. "Claude Opus 4.7"),
// which is longer than a raw model id, so the inline button gets a wider cap
// before truncating. The full string still shows in the title + popover.
const ACP_MODEL_LABEL_MAX_CHARS = 22;

function truncateModelLabel(
  model: string,
  maxChars: number = MODEL_LABEL_MAX_CHARS,
): string {
  if (model.length <= maxChars) {
    return model;
  }
  return `${model.slice(0, maxChars)}…`;
}

interface ChatInputModelMenuContentProps {
  model: ChatInputModelState;
  onClose: () => void;
  dividerInset?: "menu";
  settingsLinkClassName?: string;
  settingsIconClassName?: string;
}

export function ChatInputModelMenuContent({
  model,
  onClose,
  dividerInset,
  settingsLinkClassName,
  settingsIconClassName,
}: ChatInputModelMenuContentProps) {
  const { t } = useTranslation("openhands");
  const switchAcpModel = useSwitchAcpModel();
  const hasModelRows = model.showAcpPicker || Boolean(model.displayModel);

  const handleSelectAcpModel = (modelId: string) => {
    if (modelId !== model.currentModelId) {
      switchAcpModel.mutate({
        conversationId: model.switchConversationId,
        model: modelId,
      });
    }
    onClose();
  };

  return (
    <>
      {model.showAcpPicker ? (
        <>
          {/* role="presentation" keeps this a valid <li> child of the
              ContextMenu <ul> without exposing the section label as a
              selectable menu item (the label text is still announced). */}
          <li role="presentation" className="px-2 pt-1 pb-0.5">
            <Typography.Text className="text-[11px] font-medium text-[var(--oh-text-dim)] uppercase tracking-wide leading-4">
              {t(I18nKey.MODEL$AVAILABLE_MODELS)}
            </Typography.Text>
          </li>
          {model.availableAcpModels.map((option) => {
            const isSelected = option.id === model.currentModelId;
            return (
              <ContextMenuListItem
                key={option.id}
                testId={`chat-input-acp-model-option-${option.id}`}
                onClick={(event) => {
                  event.preventDefault();
                  event.stopPropagation();
                  handleSelectAcpModel(option.id);
                }}
                className={cn(
                  "flex items-center gap-2",
                  isSelected && "bg-[var(--oh-interactive-hover)]",
                )}
              >
                <span
                  className="flex-1 truncate text-sm leading-5"
                  title={option.label}
                >
                  {option.label}
                </span>
                {isSelected && (
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
        </>
      ) : model.displayModel ? (
        <li className="text-sm">
          <div className="p-2 leading-5 text-[var(--oh-foreground)] break-all">
            {model.displayModel}
          </div>
        </li>
      ) : null}
      {hasModelRows && <Divider inset={dividerInset} />}
      <li className="text-sm">
        <NavigationLink
          to={model.destinationPath}
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
          <span>{model.destinationLabel}</span>
        </NavigationLink>
      </li>
    </>
  );
}

export function ChatInputModel() {
  const model = useChatInputModelState();
  const [isPopoverOpen, setIsPopoverOpen] = React.useState(false);
  const triggerRef = React.useRef<HTMLButtonElement>(null);
  const popoverRef = useClickOutsideElement<HTMLUListElement>(
    () => setIsPopoverOpen(false),
    triggerRef,
  );

  if (!model.displayModel) {
    return null;
  }

  const truncatedModelLabel = truncateModelLabel(
    model.displayModel,
    model.isAcpContext ? ACP_MODEL_LABEL_MAX_CHARS : MODEL_LABEL_MAX_CHARS,
  );

  return (
    <div className="relative min-w-0">
      <button
        ref={triggerRef}
        type="button"
        className={chatInputPillButtonClassName}
        title={model.displayModel}
        data-testid="chat-input-llm-model"
        aria-expanded={isPopoverOpen}
        aria-haspopup="dialog"
        onClick={(event) => {
          event.preventDefault();
          event.stopPropagation();
          setIsPopoverOpen((open) => !open);
        }}
      >
        <span>{truncatedModelLabel}</span>
        <ComboboxCaretInline isOpen={isPopoverOpen} />
      </button>

      {isPopoverOpen && (
        <ContextMenu
          ref={popoverRef}
          testId="chat-input-llm-model-popover"
          position="top"
          alignment="left"
          spacing="none"
          className="z-[60] mb-2 min-w-[200px] max-w-[320px] max-h-[60vh] overflow-y-auto"
        >
          <ChatInputModelMenuContent
            model={model}
            onClose={() => setIsPopoverOpen(false)}
          />
        </ContextMenu>
      )}
    </div>
  );
}
