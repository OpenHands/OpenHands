import { useRef, useState } from "react";
import { Plus, X } from "lucide-react";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { OptionalTag } from "#/components/features/settings/optional-tag";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalCloseButton } from "#/components/shared/modals/modal-close-button";
import { ContextMenuListItem } from "#/components/features/context-menu/context-menu-list-item";
import { useChatInputLlmProfileState } from "#/hooks/use-chat-input-llm-profile-state";
import { useClickOutsideElement } from "#/hooks/use-click-outside-element";
import { usePromptTextareaResize } from "#/hooks/use-prompt-textarea-resize";
import { I18nKey } from "#/i18n/declaration";
import CheckIcon from "#/icons/checkmark.svg?react";
import { ComboboxCaretInline } from "#/ui/combobox-caret";
import { ContextMenu } from "#/ui/context-menu";
import { extensionModuleCardPillClassName } from "#/utils/extension-module-card-classes";
import { chatInputPillButtonClassName } from "#/utils/form-control-classes";
import { modalTitleLgClassName } from "#/utils/modal-classes";
import { cn } from "#/utils/utils";
import { formatModelNameForDisplay } from "#/utils/format-model-name";

const PROFILE_LABEL_MAX_CHARS = 18;

function truncateLabel(label: string): string {
  return label.length <= PROFILE_LABEL_MAX_CHARS
    ? label
    : `${label.slice(0, PROFILE_LABEL_MAX_CHARS)}…`;
}

function parseRepositories(value: string): string[] {
  const seen = new Set<string>();
  return value
    .split(/[\n,]+/)
    .map((entry) => entry.trim())
    .filter((entry) => {
      if (!entry || seen.has(entry)) return false;
      seen.add(entry);
      return true;
    });
}

function SetupModelPill() {
  const { t } = useTranslation("openhands");
  const profile = useChatInputLlmProfileState();
  const [isOpen, setIsOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const popoverRef = useClickOutsideElement<HTMLUListElement>(
    () => setIsOpen(false),
    triggerRef,
  );
  const displayName = profile.currentProfileName;
  const pillLabel =
    displayName ?? t(I18nKey.AUTOMATION_SETUP$MODEL_PLACEHOLDER);
  const canOpen = profile.profiles.length > 0;

  return (
    <div className="relative min-w-0">
      <button
        ref={triggerRef}
        type="button"
        className={cn(chatInputPillButtonClassName, "max-w-[200px]")}
        title={displayName ?? undefined}
        data-testid="automation-setup-model"
        aria-expanded={isOpen}
        aria-haspopup="dialog"
        onMouseDown={(event) => event.preventDefault()}
        onClick={(event) => {
          event.preventDefault();
          event.stopPropagation();
          if (!canOpen) return;
          setIsOpen((open) => !open);
        }}
      >
        <span className="truncate">
          {displayName ? truncateLabel(pillLabel) : pillLabel}
        </span>
        <ComboboxCaretInline isOpen={isOpen} />
      </button>
      {isOpen && canOpen ? (
        <ContextMenu
          ref={popoverRef}
          testId="automation-setup-model-popover"
          position="top"
          alignment="left"
          spacing="none"
          className="z-[60] mb-2 min-w-[200px] max-w-[320px] max-h-[60vh] overflow-y-auto"
        >
          {profile.profiles.map((item) => {
            const isSelected = item.name === profile.currentProfileName;
            const displayModel = formatModelNameForDisplay(item.model);
            return (
              <ContextMenuListItem
                key={item.name}
                testId={`automation-setup-model-option-${item.name}`}
                onClick={(event) => {
                  event.preventDefault();
                  event.stopPropagation();
                  profile.selectProfile(item.name);
                  setIsOpen(false);
                }}
                className={cn(
                  "flex flex-col items-stretch gap-0.5",
                  isSelected && "bg-[var(--oh-interactive-hover)]",
                )}
              >
                <div className="flex min-w-0 items-center justify-between gap-2">
                  <span className="truncate">{item.name}</span>
                  {isSelected ? (
                    <CheckIcon className="size-4 shrink-0" aria-hidden />
                  ) : null}
                </div>
                {displayModel ? (
                  <span className="truncate text-xs leading-4 text-[var(--oh-muted)]">
                    {displayModel}
                  </span>
                ) : null}
              </ContextMenuListItem>
            );
          })}
        </ContextMenu>
      ) : null}
    </div>
  );
}

function AddRepositoryModal({
  isOpen,
  onClose,
  onAdd,
}: {
  isOpen: boolean;
  onClose: () => void;
  onAdd: (address: string) => void;
}) {
  const { t } = useTranslation("openhands");
  const [address, setAddress] = useState("");
  const title = `${t(I18nKey.BUTTON$ADD)} ${t(I18nKey.AUTOMATIONS$DETAIL$REPOSITORIES)}`;

  if (!isOpen) return null;

  const trimmedAddress = address.trim();

  return (
    <ModalBackdrop onClose={onClose} aria-label={title}>
      <form
        onSubmit={(event) => {
          event.preventDefault();
          if (!trimmedAddress) return;
          onAdd(trimmedAddress);
          setAddress("");
          onClose();
        }}
        data-testid="automation-setup-add-repository-modal"
        className="relative flex w-[520px] max-w-[90vw] max-h-[85vh] flex-col rounded-xl border border-[var(--oh-border)] bg-base-secondary"
      >
        <ModalCloseButton
          onClose={onClose}
          testId="automation-setup-add-repository-modal-close"
        />
        <header className="flex-shrink-0 px-6 pb-4 pt-6">
          <h2 className={cn("pr-6", modalTitleLgClassName)}>{title}</h2>
        </header>
        <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto px-6 custom-scrollbar">
          <SettingsInput
            testId="automation-setup-repository-address"
            label={t(I18nKey.CONVERSATION$REPOSITORY)}
            type="text"
            value={address}
            onChange={setAddress}
            placeholder={t(I18nKey.SETUP$REPOSITORY_PLACEHOLDER)}
            showRequiredTag
          />
        </div>
        <footer className="flex flex-shrink-0 justify-end gap-2 px-6 pb-6 pt-4">
          <BrandButton
            type="button"
            variant="secondary"
            onClick={onClose}
            testId="automation-setup-add-repository-modal-dismiss"
          >
            {t(I18nKey.BUTTON$CLOSE)}
          </BrandButton>
          <BrandButton
            type="submit"
            variant="primary"
            testId="automation-setup-repository-submit"
            isDisabled={!trimmedAddress}
          >
            {t(I18nKey.BUTTON$ADD)}
          </BrandButton>
        </footer>
      </form>
    </ModalBackdrop>
  );
}

export function AutomationSetupPromptStack({
  prompt,
  repository,
  updatedSuffix,
  repositorySuffix,
  isStreaming,
  errorText,
  onPromptChange,
  onRepositoryChange,
}: {
  prompt: string;
  repository: string;
  updatedSuffix?: string;
  repositorySuffix?: string;
  isStreaming: boolean;
  errorText?: string;
  onPromptChange: (value: string) => void;
  onRepositoryChange: (value: string) => void;
}) {
  const { t } = useTranslation("openhands");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const { gripRef, isGripDragging, handleGripMouseDown, handleGripTouchStart } =
    usePromptTextareaResize(textareaRef);
  const [isRepositoryModalOpen, setIsRepositoryModalOpen] = useState(false);
  const repositories = parseRepositories(repository);
  const addLabel = `${t(I18nKey.BUTTON$ADD)} ${t(I18nKey.AUTOMATIONS$DETAIL$REPOSITORIES)}`;

  const writeRepositories = (next: string[]) => {
    onRepositoryChange(next.join(", "));
  };

  return (
    <label
      data-streaming-active={isStreaming ? "true" : undefined}
      className="flex w-full min-w-0 flex-col gap-2.5"
    >
      <span className="flex items-center gap-2 text-sm">
        {t(I18nKey.AUTOMATIONS$PROMPT)}
        {updatedSuffix ? (
          <span className="font-normal text-[var(--oh-muted)]">
            {updatedSuffix}
          </span>
        ) : null}
      </span>
      <div
        data-testid="automation-setup-prompt-stack"
        className="relative w-full"
      >
        <div
          data-testid="automation-setup-prompt-container"
          className="relative z-10 -mb-[15px] flex flex-col rounded-[15px] border border-[var(--oh-border)] bg-[var(--oh-surface)] p-4"
        >
          <textarea
            ref={textareaRef}
            data-testid="automation-setup-prompt"
            name="prompt"
            value={prompt}
            onChange={(event) => onPromptChange(event.target.value)}
            placeholder={t(I18nKey.HOME$AUTOMATE_PROMPT_PLACEHOLDER)}
            rows={5}
            className="min-h-[120px] w-full resize-none border-0 bg-transparent p-0 text-sm text-content outline-none placeholder:text-tertiary-alt placeholder:italic"
          />
          <div className="flex min-w-0 items-center pt-2">
            <SetupModelPill />
          </div>
          <div
            data-testid="automation-setup-prompt-grip"
            className="group absolute bottom-0 left-0 z-20 h-3 w-full"
          >
            <div
              className="absolute inset-0 z-[1] cursor-ns-resize select-none"
              onMouseDown={handleGripMouseDown}
              onTouchStart={handleGripTouchStart}
              aria-hidden
            />
            <div
              ref={gripRef}
              className={cn(
                "pointer-events-none absolute bottom-0 left-0 z-[2] h-px w-full bg-white transition-opacity duration-200",
                isGripDragging
                  ? "opacity-100"
                  : "opacity-0 group-hover:opacity-100",
              )}
            />
          </div>
        </div>
        <div
          data-testid="automation-setup-prompt-drawer"
          className="flex min-h-9 items-center rounded-b-[15px] bg-[var(--oh-surface-raised)] px-4 pb-3 pt-[calc(15px+0.5rem)]"
        >
          <div
            data-testid="automation-setup-repository"
            className="flex w-full min-w-0 items-center gap-1 overflow-x-auto"
          >
            <div className="flex shrink-0 items-center gap-2">
              <span className="text-sm">
                {t(I18nKey.AUTOMATIONS$DETAIL$REPOSITORIES)}
              </span>
              <OptionalTag />
              {repositorySuffix ? (
                <span className="text-xs text-[var(--oh-muted)]">
                  {repositorySuffix}
                </span>
              ) : null}
              <button
                type="button"
                data-testid="automation-setup-repository-add"
                aria-label={addLabel}
                className="inline-flex size-6 shrink-0 items-center justify-center rounded-md text-[var(--oh-muted)] hover:bg-white/10 hover:text-white"
                onMouseDown={(event) => event.preventDefault()}
                onClick={() => setIsRepositoryModalOpen(true)}
              >
                <Plus className="size-4" aria-hidden />
              </button>
            </div>
            {repositories.map((item) => (
              <span
                key={item}
                data-testid="automation-setup-repository-value"
                className={cn(extensionModuleCardPillClassName, "gap-1 pr-1")}
              >
                <span className="truncate">{item}</span>
                <button
                  type="button"
                  data-testid="automation-setup-repository-remove"
                  aria-label={`${t(I18nKey.COMMON$REMOVE)} ${item}`}
                  className="inline-flex size-4 items-center justify-center rounded-full text-tertiary-light hover:bg-white/10 hover:text-white"
                  onMouseDown={(event) => event.preventDefault()}
                  onClick={() =>
                    writeRepositories(
                      repositories.filter((entry) => entry !== item),
                    )
                  }
                >
                  <X className="size-3" aria-hidden />
                </button>
              </span>
            ))}
          </div>
        </div>
      </div>
      {errorText ? (
        <span
          role="alert"
          className="text-xs leading-5 text-[var(--oh-warning)]"
        >
          {errorText}
        </span>
      ) : null}
      <AddRepositoryModal
        isOpen={isRepositoryModalOpen}
        onClose={() => setIsRepositoryModalOpen(false)}
        onAdd={(address) => {
          if (repositories.includes(address)) return;
          writeRepositories([...repositories, address]);
        }}
      />
    </label>
  );
}
