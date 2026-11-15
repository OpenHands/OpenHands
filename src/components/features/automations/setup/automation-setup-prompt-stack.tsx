import {
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type ReactNode,
} from "react";
import { Info, Lock, Plus, X } from "lucide-react";
import { useTranslation } from "react-i18next";
import { getActiveBackend } from "#/api/backend-registry/active-store";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalCloseButton } from "#/components/shared/modals/modal-close-button";
import { ContextMenuListItem } from "#/components/features/context-menu/context-menu-list-item";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import { useGitRepositories } from "#/hooks/query/use-git-repositories";
import { useSearchRepositories } from "#/hooks/query/use-search-repositories";
import { useLlmProfiles } from "#/hooks/query/use-llm-profiles";
import { useAgentProfiles } from "#/hooks/query/use-agent-profiles";
import { useClickOutsideElement } from "#/hooks/use-click-outside-element";
import { useCloseOnEscape } from "#/hooks/use-close-on-escape";
import { useDebounce } from "#/hooks/use-debounce";
import { useUserProviders } from "#/hooks/use-user-providers";
import { useDeploymentCapabilities } from "#/hooks/query/use-manifest-capabilities";
import { useSettings } from "#/hooks/query/use-settings";
import { usePromptTextareaResize } from "#/hooks/use-prompt-textarea-resize";
import { I18nKey } from "#/i18n/declaration";
import CheckIcon from "#/icons/checkmark.svg?react";
import SearchIcon from "#/icons/search.svg?react";
import { ComboboxCaretInline } from "#/ui/combobox-caret";
import { Divider } from "#/ui/divider";
import { ContextMenu } from "#/ui/context-menu";
import { extensionModuleCardPillClassName } from "#/utils/extension-module-card-classes";
import { chatInputPillButtonClassName } from "#/utils/form-control-classes";
import { modalTitleLgClassName } from "#/utils/modal-classes";
import { cn, getProviderName } from "#/utils/utils";
import { formatModelNameForDisplay } from "#/utils/format-model-name";
import {
  automationCloneProviders,
  automationRepositoryAddress,
  providerHostKnown,
} from "#/utils/automation-repo-sources";
import type { Provider } from "#/types/settings";

const PROFILE_LABEL_MAX_CHARS = 18;
const REPOSITORY_SEARCH_DEBOUNCE_MS = 300;

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

// Typing in the repository menu stays inside it, except Escape, which closes
// the menu through `useCloseOnEscape`.
function stopMenuKeyPropagation(event: ReactKeyboardEvent) {
  if (event.key !== "Escape") event.stopPropagation();
}

function ProfileMenuItem({
  testId,
  label,
  detail,
  isSelected,
  onSelect,
}: {
  testId: string;
  label: string;
  detail?: string | null;
  isSelected: boolean;
  onSelect: () => void;
}) {
  return (
    <ContextMenuListItem
      testId={testId}
      onClick={(event) => {
        event.preventDefault();
        event.stopPropagation();
        onSelect();
      }}
      className={cn(
        "flex flex-col items-stretch gap-0.5",
        isSelected && "bg-interactive-hover",
      )}
    >
      <div className="flex min-w-0 items-center justify-between gap-2">
        <span className="truncate">{label}</span>
        {isSelected ? (
          <CheckIcon className="size-4 shrink-0" aria-hidden />
        ) : null}
      </div>
      {detail ? (
        <span className="truncate text-xs leading-4 text-muted">{detail}</span>
      ) : null}
    </ContextMenuListItem>
  );
}

/** A pill that opens a menu of profiles; the shape both profile pickers share. */
function ProfilePill({
  testId,
  label,
  tooltip,
  canOpen,
  lockedReason,
  children,
}: {
  testId: string;
  label: string;
  tooltip: string;
  canOpen: boolean;
  /**
   * Why the pill is shown but not choosable, because another setting decides
   * it. The reason is the lock icon's tooltip.
   */
  lockedReason?: string;
  children: (close: () => void) => ReactNode;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const popoverRef = useClickOutsideElement<HTMLUListElement>(
    () => setIsOpen(false),
    triggerRef,
  );
  useCloseOnEscape(isOpen, () => setIsOpen(false), triggerRef);
  const close = () => setIsOpen(false);
  const locked = lockedReason !== undefined;

  const trigger = (
    <button
      ref={triggerRef}
      type="button"
      className={cn(
        chatInputPillButtonClassName,
        "automation-setup-model-trigger rounded-lg",
        locked && "cursor-not-allowed border-dashed text-muted",
      )}
      data-testid={testId}
      aria-label={label}
      aria-disabled={locked || undefined}
      aria-expanded={locked ? undefined : isOpen}
      aria-haspopup={locked ? undefined : "dialog"}
      onMouseDown={(event) => event.preventDefault()}
      onClick={(event) => {
        event.preventDefault();
        event.stopPropagation();
        if (locked || !canOpen) return;
        setIsOpen((open) => !open);
      }}
    >
      <span className="truncate">{truncateLabel(label)}</span>
      {locked ? null : <ComboboxCaretInline isOpen={isOpen} />}
    </button>
  );

  return (
    <div className="relative flex min-w-0 items-center gap-1.5">
      {locked ? (
        <>
          {trigger}
          <StyledTooltip content={lockedReason} placement="top">
            {/* A button so keyboard users can reach the explanation too. */}
            <button
              type="button"
              aria-label={lockedReason}
              data-testid={`${testId}-lock`}
              className="inline-flex shrink-0 cursor-help rounded text-muted hover:text-content focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"
            >
              <Lock className="size-3.5" aria-hidden />
            </button>
          </StyledTooltip>
        </>
      ) : (
        <StyledTooltip content={tooltip} placement="top">
          {trigger}
        </StyledTooltip>
      )}
      {isOpen && canOpen && !locked ? (
        <ContextMenu
          ref={popoverRef}
          testId={`${testId}-popover`}
          position="top"
          alignment="left"
          spacing="none"
          className="automation-setup-model-popover z-60 mb-2 overflow-y-auto"
        >
          {children(close)}
        </ContextMenu>
      ) : null}
    </div>
  );
}

export function SetupModelPill({
  value,
  onChange,
  lockedTo,
}: {
  value: string;
  onChange: (profileName: string) => void;
  /**
   * The LLM profile an agent profile runs on. The service refuses a model
   * alongside an agent profile, so the pill shows this one and can't change.
   */
  lockedTo?: { llmProfile: string | null; reason: string };
}) {
  const { t } = useTranslation("openhands");
  const { data, isLoading } = useLlmProfiles();
  const profiles = data?.profiles ?? [];
  const activeProfileName = data?.active_profile ?? null;
  const pinnedProfileName = value.trim() || null;
  const label = lockedTo
    ? (lockedTo.llmProfile ??
      activeProfileName ??
      t(I18nKey.COMMON$ACTIVE_PROFILE))
    : (pinnedProfileName ??
      activeProfileName ??
      t(I18nKey.AUTOMATION_SETUP$MODEL_PLACEHOLDER));

  return (
    <ProfilePill
      testId="automation-setup-model"
      label={label}
      tooltip={t(I18nKey.LLM$MODEL)}
      canOpen={!isLoading && profiles.length > 0}
      lockedReason={lockedTo?.reason}
    >
      {(close) => (
        <>
          <ProfileMenuItem
            testId="automation-setup-model-option-active"
            label={t(I18nKey.COMMON$ACTIVE_PROFILE)}
            isSelected={!pinnedProfileName}
            onSelect={() => {
              onChange("");
              close();
            }}
          />
          {profiles.map((item) => (
            <ProfileMenuItem
              key={item.name}
              testId={`automation-setup-model-option-${item.name}`}
              label={item.name}
              detail={formatModelNameForDisplay(item.model)}
              isSelected={item.name === pinnedProfileName}
              onSelect={() => {
                onChange(item.name);
                close();
              }}
            />
          ))}
        </>
      )}
    </ProfilePill>
  );
}

/**
 * Picks the agent profile an automation runs from, or None. Each option names
 * the LLM profile that agent profile would run on.
 */
export function SetupAgentProfilePill({
  value,
  onChange,
}: {
  value: string;
  onChange: (agentProfileId: string) => void;
}) {
  const { t } = useTranslation("openhands");
  const { data, isLoading } = useAgentProfiles();
  const { data: llmProfiles } = useLlmProfiles();
  const activeLlmProfile = llmProfiles?.active_profile ?? null;
  const profiles = (data?.profiles ?? []).filter(
    (profile): profile is typeof profile & { id: string } =>
      Boolean(profile.id),
  );
  const selectedId = value.trim() || null;
  const selected = profiles.find((profile) => profile.id === selectedId);
  let label = t(I18nKey.AUTOMATION_SETUP$AGENT_PROFILE_NONE);
  if (selected) label = selected.name;
  else if (selectedId)
    label = t(I18nKey.AUTOMATIONS$UNAVAILABLE_AGENT_PROFILE, {
      id: selectedId,
    });

  return (
    <ProfilePill
      testId="automation-setup-agent-profile"
      label={label}
      tooltip={t(I18nKey.CHAT$AGENT_PROFILE_PLACEHOLDER)}
      canOpen={!isLoading && profiles.length > 0}
    >
      {(close) => (
        <>
          <ProfileMenuItem
            testId="automation-setup-agent-profile-option-none"
            label={t(I18nKey.AUTOMATION_SETUP$AGENT_PROFILE_NONE)}
            isSelected={!selectedId}
            onSelect={() => {
              onChange("");
              close();
            }}
          />
          {profiles.map((item) => (
            <ProfileMenuItem
              key={item.id}
              testId={`automation-setup-agent-profile-option-${item.id}`}
              label={item.name}
              detail={
                item.agent_kind === "acp"
                  ? t(I18nKey.CONVERSATION$ACP_AGENT_GENERIC)
                  : item.llm_profile_ref || activeLlmProfile
              }
              isSelected={item.id === selectedId}
              onSelect={() => {
                onChange(item.id);
                close();
              }}
            />
          ))}
        </>
      )}
    </ProfilePill>
  );
}

/**
 * The service refuses an LLM profile next to an agent profile, so while one is
 * chosen the LLM profile pill shows what it runs on and stays locked. An ACP
 * agent runs its own model, not an LLM profile.
 */
function useModelLockedByAgentProfile(agentProfileId: string) {
  const { t } = useTranslation("openhands");
  const { data } = useAgentProfiles();
  const selectedId = agentProfileId.trim() || null;
  if (!selectedId) return undefined;
  const selected = data?.profiles?.find((profile) => profile.id === selectedId);
  return {
    llmProfile:
      selected?.agent_kind === "acp"
        ? t(I18nKey.CONVERSATION$ACP_AGENT_GENERIC)
        : selected?.llm_profile_ref || null,
    reason: t(I18nKey.AUTOMATION_SETUP$LLM_PROFILE_SET_BY_AGENT, {
      name: selected?.name ?? selectedId,
    }),
  };
}

/**
 * What a script automation runs with: an agent profile, or an LLM profile;
 * None hands the choice back to the LLM profile.
 */
export function SetupProfilePickers({
  model,
  onModelChange,
  agentProfileId,
  onAgentProfileChange,
}: {
  model: string;
  onModelChange: (profileName: string) => void;
  agentProfileId: string;
  onAgentProfileChange: (agentProfileId: string) => void;
}) {
  const { t } = useTranslation("openhands");
  const modelLock = useModelLockedByAgentProfile(agentProfileId);

  return (
    <div
      data-testid="automation-setup-profile-pickers"
      className="flex flex-wrap items-start gap-x-6 gap-y-3"
    >
      <div className="flex min-w-0 flex-col gap-1.5">
        <span className="text-xs text-muted">
          {t(I18nKey.CHAT$AGENT_PROFILE_PLACEHOLDER)}
        </span>
        <SetupAgentProfilePill
          value={agentProfileId}
          onChange={(value) => {
            // A model would be refused next to the profile; drop it.
            if (value) onModelChange("");
            onAgentProfileChange(value);
          }}
        />
      </div>
      <div className="flex min-w-0 flex-col gap-1.5">
        <span className="text-xs text-muted">
          {t(I18nKey.AUTOMATIONS$DETAIL$MODEL)}
        </span>
        <SetupModelPill
          value={model}
          onChange={onModelChange}
          lockedTo={modelLock}
        />
      </div>
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
        className="automation-setup-repository-dialog relative flex flex-col rounded-xl border border-border bg-base-secondary"
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
  titleAction,
  showTitle = true,
  model,
  agentProfileId = "",
  repositoryReadOnly = false,
  onPromptChange,
  onRepositoryChange,
  onModelChange,
  onAgentProfileChange,
}: {
  prompt: string;
  repository: string;
  updatedSuffix?: string;
  repositorySuffix?: string;
  isStreaming: boolean;
  errorText?: string;
  /** Sits on the Prompt title row, outside the field label. */
  titleAction?: ReactNode;
  /** The setup form owns the title when the kind toggle stays outside the crossfade. */
  showTitle?: boolean;
  model: string;
  agentProfileId?: string;
  /** Show the repositories without letting them be added or removed. */
  repositoryReadOnly?: boolean;
  onPromptChange: (value: string) => void;
  onRepositoryChange: (value: string) => void;
  onModelChange: (profileName: string) => void;
  /** Offers the agent profile picker; without it only the model is offered. */
  onAgentProfileChange?: (agentProfileId: string) => void;
}) {
  const { t } = useTranslation("openhands");
  const modelLock = useModelLockedByAgentProfile(agentProfileId);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const { gripRef, isGripDragging, handleGripMouseDown, handleGripTouchStart } =
    usePromptTextareaResize(textareaRef, { contentKey: prompt });
  const [isRepositoryModalOpen, setIsRepositoryModalOpen] = useState(false);
  const [isRepositoryMenuOpen, setIsRepositoryMenuOpen] = useState(false);
  const [repositorySearch, setRepositorySearch] = useState("");
  const repositoryAddRef = useRef<HTMLButtonElement>(null);
  const repositoryMenuRef = useClickOutsideElement<HTMLDivElement>(
    () => setIsRepositoryMenuOpen(false),
    repositoryAddRef,
  );
  useCloseOnEscape(
    isRepositoryMenuOpen,
    () => setIsRepositoryMenuOpen(false),
    repositoryAddRef,
  );
  const isLocalBackend = getActiveBackend().backend.kind === "local";
  const { providers } = useUserProviders();
  const { data: settings } = useSettings();
  const providerHosts = settings?.provider_tokens_set ?? {};
  const deploymentCapabilities = useDeploymentCapabilities();
  // The service says whose repositories its runs clone; older services predate
  // the field and clone from the three hosted providers.
  const cloneProviders = automationCloneProviders(
    deploymentCapabilities.data?.repoProviders,
  );
  // Whose repositories the picker can offer: a connected provider the runs
  // clone from, whose host is known (a GitLab connection needs the server to
  // report its host, or the form could not tell gitlab.com from an instance).
  const canListProvider = (provider: Provider) =>
    cloneProviders.includes(provider) &&
    providerHostKnown(provider, providerHosts[provider]);
  const connectedCloneProviders = providers.filter(canListProvider);
  const [selectedProvider, setSelectedProvider] = useState<Provider | null>(
    null,
  );
  const defaultProvider: Provider | null = connectedCloneProviders.includes(
    "github",
  )
    ? "github"
    : (connectedCloneProviders[0] ??
      providers[0] ??
      (isLocalBackend ? "github" : null));
  const repositoryProvider: Provider | null =
    selectedProvider && connectedCloneProviders.includes(selectedProvider)
      ? selectedProvider
      : defaultProvider;
  // What the form stores for a listed repository: GitHub's shorthand, or the
  // full URL on the provider's host for anything else.
  const addressFor = (name: string): string | null =>
    repositoryProvider
      ? automationRepositoryAddress(
          repositoryProvider,
          name,
          providerHosts[repositoryProvider],
        )
      : name;
  const repositoryQuery = useGitRepositories({
    provider: repositoryProvider,
    enabled: isRepositoryMenuOpen && repositoryProvider !== null,
    allowMissingProvider: isLocalBackend,
  });
  const repositories = parseRepositories(repository);
  const repositoryPages = repositoryQuery.data?.pages ?? [];
  const isLocalGithubTokenMissing =
    isLocalBackend && repositoryPages.some((page) => page.missing_token);
  const isRepositoryListLoading = repositoryQuery.isLoading;
  const isRepositoryListError = repositoryQuery.isError;
  const repositoryNames = repositoryPages
    .flatMap((page) => page.items)
    .map((repo) => repo.full_name);
  const repositorySearchText = repositorySearch.trim().toLowerCase();
  // The loaded pages may not hold every match, so a search also asks the
  // provider.
  const debouncedRepositorySearch = useDebounce(
    repositorySearch.trim(),
    REPOSITORY_SEARCH_DEBOUNCE_MS,
  );
  const repositorySearchQuery = useSearchRepositories(
    debouncedRepositorySearch,
    repositoryProvider,
    !isRepositoryMenuOpen,
  );
  const isRepositorySearchLoading =
    repositorySearchText !== "" &&
    (repositorySearchQuery.isFetching ||
      debouncedRepositorySearch !== repositorySearch.trim());
  const searchedRepositoryNames = repositorySearchText
    ? (repositorySearchQuery.data ?? []).map((repo) => repo.full_name)
    : [];
  const listedRepositoryNames = [
    ...repositoryNames,
    ...searchedRepositoryNames,
  ].filter((name, index, names) => {
    const address = addressFor(name);
    return (
      (address === null || !repositories.includes(address)) &&
      names.indexOf(name) === index
    );
  });
  const visibleRepositoryNames = repositorySearchText
    ? listedRepositoryNames.filter((name) =>
        name.toLowerCase().includes(repositorySearchText),
      )
    : listedRepositoryNames;
  const addLabel = `${t(I18nKey.BUTTON$ADD)} ${t(I18nKey.AUTOMATIONS$DETAIL$REPOSITORIES)}`;
  const isRepositoryProviderUnsupported =
    repositoryProvider !== null && !canListProvider(repositoryProvider);

  const writeRepositories = (next: string[]) => {
    onRepositoryChange(next.join(", "));
  };

  return (
    <div
      className={cn(
        "flex w-full min-w-0 flex-col gap-2.5",
        isRepositoryMenuOpen && "relative z-30",
      )}
    >
      {showTitle ? (
        <div className="flex w-full items-center gap-2">
          <span className="flex items-center gap-2 text-sm">
            {t(I18nKey.AUTOMATIONS$PROMPT)}
            {updatedSuffix ? (
              <span className="font-normal text-muted">{updatedSuffix}</span>
            ) : null}
          </span>
          {titleAction ? (
            <div className="ml-auto shrink-0">{titleAction}</div>
          ) : null}
        </div>
      ) : null}
      <label
        data-streaming-active={isStreaming ? "true" : undefined}
        className="flex w-full min-w-0 flex-col gap-2.5"
      >
        <div
          data-testid="automation-setup-prompt-stack"
          className="relative w-full"
        >
          <div
            data-testid="automation-setup-prompt-container"
            className="automation-setup-joined-card relative z-10 flex flex-col border border-border bg-surface p-4"
          >
            <textarea
              ref={textareaRef}
              data-testid="automation-setup-prompt"
              name="prompt"
              value={prompt}
              onChange={(event) => onPromptChange(event.target.value)}
              placeholder={t(I18nKey.HOME$AUTOMATE_PROMPT_PLACEHOLDER)}
              rows={5}
              className="automation-setup-prompt-input w-full resize-none border-0 bg-transparent p-0 text-sm text-content outline-none placeholder:text-tertiary-alt"
            />
            <div className="flex min-w-0 items-center gap-2 pt-2">
              {onAgentProfileChange ? (
                <SetupAgentProfilePill
                  value={agentProfileId}
                  onChange={(value) => {
                    if (value) onModelChange("");
                    onAgentProfileChange(value);
                  }}
                />
              ) : null}
              <SetupModelPill
                value={model}
                onChange={onModelChange}
                lockedTo={onAgentProfileChange ? modelLock : undefined}
              />
            </div>
            <div
              data-testid="automation-setup-prompt-grip"
              className="group absolute bottom-0 left-0 z-20 h-3 w-full"
            >
              <div
                className="absolute inset-0 z-1 cursor-ns-resize select-none"
                onMouseDown={handleGripMouseDown}
                onTouchStart={handleGripTouchStart}
                aria-hidden
              />
              <div
                ref={gripRef}
                className={cn(
                  "pointer-events-none absolute bottom-0 left-0 z-2 h-px w-full bg-white transition-opacity duration-200",
                  isGripDragging
                    ? "opacity-100"
                    : "opacity-0 group-hover:opacity-100",
                )}
              />
            </div>
          </div>
          <div
            data-testid="automation-setup-prompt-drawer"
            className="automation-setup-joined-tray flex min-h-9 items-center border-x border-b border-border bg-surface-raised px-4 pb-2"
          >
            <div
              data-testid="automation-setup-repository"
              className="flex w-full min-w-0 items-start gap-1"
            >
              <div className="relative flex shrink-0 items-center gap-2">
                <span className="text-sm">
                  {t(I18nKey.AUTOMATIONS$DETAIL$REPOSITORIES)}
                </span>
                {repositorySuffix ? (
                  <span className="text-xs text-muted">{repositorySuffix}</span>
                ) : null}
                {repositoryReadOnly ? (
                  <span
                    data-testid="automation-setup-repository-read-only"
                    className="text-xs text-muted"
                  >
                    {t(I18nKey.AUTOMATION_SETUP$REPOSITORIES_READ_ONLY)}
                  </span>
                ) : (
                  <button
                    ref={repositoryAddRef}
                    type="button"
                    data-testid="automation-setup-repository-add"
                    aria-label={addLabel}
                    aria-haspopup="menu"
                    aria-expanded={isRepositoryMenuOpen}
                    className="inline-flex size-6 shrink-0 items-center justify-center rounded-md text-muted hover:bg-white/10 hover:text-white"
                    onMouseDown={(event) => event.preventDefault()}
                    onClick={() => {
                      setRepositorySearch("");
                      setIsRepositoryMenuOpen((open) => !open);
                    }}
                  >
                    <Plus className="size-4" aria-hidden />
                  </button>
                )}
                {isLocalGithubTokenMissing && !repositoryReadOnly ? (
                  <StyledTooltip
                    content={t(
                      I18nKey.AUTOMATION_SETUP$GITHUB_TOKEN_REQUIRED_TOOLTIP,
                    )}
                    placement="top"
                  >
                    <button
                      type="button"
                      aria-label={t(
                        I18nKey.AUTOMATION_SETUP$GITHUB_TOKEN_REQUIRED_TOOLTIP,
                      )}
                      data-testid="automation-setup-repository-token-tooltip"
                      className="inline-flex size-5 shrink-0 items-center justify-center rounded-full text-muted"
                      onMouseDown={(event) => event.preventDefault()}
                    >
                      <Info className="size-4" aria-hidden />
                    </button>
                  </StyledTooltip>
                ) : null}
                {isRepositoryMenuOpen ? (
                  <div
                    ref={repositoryMenuRef}
                    role="menu"
                    data-testid="automation-setup-repository-menu"
                    className="automation-setup-repository-menu absolute top-full left-0 z-60 mt-1 flex max-h-72 flex-col overflow-hidden rounded-md border border-border-subtle bg-tertiary py-1 shadow-lg"
                  >
                    <div className="shrink-0 px-2">
                      <div className="relative">
                        <SearchIcon
                          width={16}
                          height={16}
                          aria-hidden
                          className="pointer-events-none absolute top-1/2 left-0 -translate-y-1/2 text-muted"
                        />
                        <input
                          type="text"
                          data-testid="automation-setup-repository-search"
                          value={repositorySearch}
                          onChange={(event) =>
                            setRepositorySearch(event.target.value)
                          }
                          onKeyDown={stopMenuKeyPropagation}
                          placeholder={t(I18nKey.COMMON$SEARCH_REPOSITORIES)}
                          aria-label={t(I18nKey.COMMON$SEARCH_REPOSITORIES)}
                          className="w-full border-0 bg-transparent py-1.5 pr-0 pl-6 text-sm text-contrast outline-none placeholder:text-muted focus:ring-0 focus:outline-none"
                        />
                      </div>
                    </div>
                    {connectedCloneProviders.length > 1 ? (
                      <div className="shrink-0 px-2 pb-1">
                        <select
                          data-testid="automation-setup-repository-provider"
                          aria-label={t(I18nKey.CONVERSATION$GIT_PROVIDER)}
                          value={repositoryProvider ?? ""}
                          onChange={(event) => {
                            setSelectedProvider(event.target.value as Provider);
                            setRepositorySearch("");
                          }}
                          onKeyDown={stopMenuKeyPropagation}
                          className="w-full rounded-md border border-border-subtle bg-tertiary px-2 py-1 text-sm text-content outline-none"
                        >
                          {connectedCloneProviders.map((provider) => (
                            <option key={provider} value={provider}>
                              {getProviderName(provider)}
                            </option>
                          ))}
                        </select>
                      </div>
                    ) : null}
                    <Divider inset="menu" />
                    <div
                      className="min-h-0 flex-1 overflow-y-auto px-1"
                      onScroll={(event) => {
                        const list = event.currentTarget;
                        if (
                          list.scrollTop + list.clientHeight >=
                          list.scrollHeight - 24
                        ) {
                          repositoryQuery.onLoadMore();
                        }
                      }}
                    >
                      {isRepositoryProviderUnsupported && repositoryProvider ? (
                        <p
                          data-testid="automation-setup-repository-unsupported"
                          className="px-2 py-2 text-sm text-muted"
                        >
                          {t(
                            I18nKey.AUTOMATION_SETUP$REPOSITORY_PROVIDER_UNSUPPORTED,
                            { provider: getProviderName(repositoryProvider) },
                          )}
                        </p>
                      ) : null}
                      {isRepositoryListError ? (
                        <p
                          data-testid="automation-setup-repository-error"
                          className="px-2 py-2 text-sm text-warning"
                        >
                          {t(I18nKey.HOME$FAILED_TO_LOAD_REPOSITORIES)}
                        </p>
                      ) : (isRepositoryListLoading ||
                          isRepositorySearchLoading) &&
                        visibleRepositoryNames.length === 0 ? (
                        <div
                          data-testid="automation-setup-repository-loading"
                          className="flex items-center justify-center py-3"
                        >
                          <LoadingSpinner size="small" />
                        </div>
                      ) : (
                        visibleRepositoryNames.map((name) => {
                          // Organization/project/repository names can be
                          // long; each option stays on one line and shortens
                          // the organization/project part first, so the
                          // repository name stays visible.
                          const nameStart = Math.max(name.lastIndexOf("/"), 0);
                          return (
                            <ContextMenuListItem
                              key={name}
                              testId={`automation-setup-repository-option-${name}`}
                              onClick={(event) => {
                                event.preventDefault();
                                event.stopPropagation();
                                const address = addressFor(name);
                                if (
                                  address &&
                                  !repositories.includes(address)
                                ) {
                                  writeRepositories([...repositories, address]);
                                }
                                setIsRepositoryMenuOpen(false);
                              }}
                              isDisabled={isRepositoryProviderUnsupported}
                              className="flex w-full items-center"
                            >
                              <span className="flex min-w-0" title={name}>
                                <span className="min-w-0 truncate">
                                  {name.slice(0, nameStart)}
                                </span>
                                <span className="max-w-full shrink-0 truncate">
                                  {name.slice(nameStart)}
                                </span>
                              </span>
                            </ContextMenuListItem>
                          );
                        })
                      )}
                      {repositorySearchText &&
                      !isRepositoryListLoading &&
                      !isRepositorySearchLoading &&
                      !isRepositoryListError &&
                      visibleRepositoryNames.length === 0 ? (
                        <p className="px-2 py-2 text-sm text-muted italic">
                          {t(I18nKey.GITHUB$NO_RESULTS)}
                        </p>
                      ) : null}
                    </div>
                    <Divider
                      inset="menu"
                      testId="automation-setup-repository-custom-divider"
                    />
                    <div className="shrink-0 px-1">
                      <ContextMenuListItem
                        testId="automation-setup-repository-custom"
                        onClick={(event) => {
                          event.preventDefault();
                          event.stopPropagation();
                          setIsRepositoryMenuOpen(false);
                          setIsRepositoryModalOpen(true);
                        }}
                        className="flex w-full items-center"
                      >
                        <span className="truncate">
                          {t(I18nKey.AUTOMATION_SETUP$TYPE_CUSTOM)}
                        </span>
                      </ContextMenuListItem>
                    </div>
                  </div>
                ) : null}
              </div>
              <div
                data-testid="automation-setup-repository-values"
                className="flex min-w-0 flex-1 flex-wrap items-center gap-1"
              >
                {repositories.map((item) => (
                  <span
                    key={item}
                    data-testid="automation-setup-repository-value"
                    className={cn(
                      extensionModuleCardPillClassName,
                      "gap-1 pr-1",
                    )}
                  >
                    <span className="truncate" title={item}>
                      {item}
                    </span>
                    {repositoryReadOnly ? null : (
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
                    )}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
        {errorText ? (
          <span role="alert" className="text-xs leading-5 text-warning">
            {errorText}
          </span>
        ) : null}
      </label>
      <AddRepositoryModal
        isOpen={isRepositoryModalOpen}
        onClose={() => setIsRepositoryModalOpen(false)}
        onAdd={(address) => {
          if (repositories.includes(address)) return;
          writeRepositories([...repositories, address]);
        }}
      />
    </div>
  );
}
