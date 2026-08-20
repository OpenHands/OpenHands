import React from "react";
import { useTranslation } from "react-i18next";
import { useCreateConversation } from "#/hooks/mutation/use-create-conversation";
// Removed useRepositoryBranches import - GitBranchDropdown manages its own data
import { useNavigation } from "#/context/navigation-context";
import { useIsCreatingConversation } from "#/hooks/use-is-creating-conversation";
import { Branch, GitRepository } from "#/types/git";
import { BrandButton } from "../settings/brand-button";
import { useUserProviders } from "#/hooks/use-user-providers";
import { Provider } from "#/types/settings";
import { I18nKey } from "#/i18n/declaration";
import RepoForkedIcon from "#/icons/repo-forked.svg?react";
import { GitProviderDropdown } from "./git-provider-dropdown";
import { GitBranchDropdown } from "./git-branch-dropdown";
import { GitRepoDropdown } from "./git-repo-dropdown";
import { useHomeStore } from "#/stores/home-store";

interface RepositorySelectionFormProps {
  /**
   * Optional callback fired whenever the user picks or clears a repository.
   * The form itself owns the "Launch" action — it creates the conversation
   * and navigates internally — so this prop is only useful for callers that
   * want to mirror the selection in their own state (e.g. to filter a
   * sibling list by the currently picked repo).
   */
  onRepoSelection?: (repo: GitRepository | null) => void;
  isLoadingSettings?: boolean;
  /**
   * When provided, the form skips its own conversation creation + navigation
   * and just calls back with the selected repo/branch/provider. Used by the
   * home launcher dialog so the user can confirm a selection without
   * immediately starting a conversation.
   */
  onConfirm?: (selection: {
    repository: GitRepository;
    branch: Branch;
    provider: Provider | null;
  }) => void;
}

export function RepositorySelectionForm({
  onRepoSelection,
  isLoadingSettings = false,
  onConfirm,
}: RepositorySelectionFormProps) {
  const { navigate } = useNavigation();

  const [selectedRepository, setSelectedRepository] =
    React.useState<GitRepository | null>(null);
  const [selectedBranch, setSelectedBranch] = React.useState<Branch | null>(
    null,
  );
  const [selectedProvider, setSelectedProvider] =
    React.useState<Provider | null>(null);

  const { providers } = useUserProviders();
  const {
    addRecentRepository,
    setLastSelectedProvider,
    getLastSelectedProvider,
  } = useHomeStore();
  const {
    mutate: createConversation,
    isPending,
    isSuccess,
  } = useCreateConversation();

  const isCreatingConversationElsewhere = useIsCreatingConversation();

  const { t } = useTranslation("openhands");

  // Auto-select provider logic
  React.useEffect(() => {
    if (providers.length === 0) return;

    // If there's only one provider, auto-select it
    if (providers.length === 1 && !selectedProvider) {
      setSelectedProvider(providers[0]);
      return;
    }

    // If there are multiple providers and none is selected, try to use the last selected one
    if (providers.length > 1 && !selectedProvider) {
      const lastSelected = getLastSelectedProvider();
      if (lastSelected && providers.includes(lastSelected)) {
        setSelectedProvider(lastSelected);
      }
    }
  }, [providers, selectedProvider, getLastSelectedProvider]);

  // We check for isSuccess because the app might require time to render
  // into the new conversation screen after the conversation is created.
  const isCreatingConversation =
    isPending || isSuccess || isCreatingConversationElsewhere;

  // Branch selection is now handled by GitBranchDropdown component

  const handleProviderSelection = (provider: Provider | null) => {
    if (provider === selectedProvider) {
      return;
    }

    setSelectedProvider(provider);
    setLastSelectedProvider(provider); // Store the selected provider
    setSelectedRepository(null); // Reset repository selection when provider changes
    setSelectedBranch(null); // Reset branch selection when provider changes
    onRepoSelection?.(null); // Reset parent component's selected repo
  };

  const handleBranchSelection = React.useCallback((branch: Branch | null) => {
    setSelectedBranch(branch);
  }, []);

  // Render the provider dropdown
  const renderProviderSelector = () => {
    // Only render if there are multiple providers
    if (providers.length <= 1) {
      return null;
    }

    return (
      <GitProviderDropdown
        providers={providers}
        value={selectedProvider}
        className="max-w-[500px]"
        onChange={handleProviderSelection}
        disabled={isLoadingSettings}
      />
    );
  };

  // Render the repository selector using our new component
  const renderRepositorySelector = () => {
    const handleRepoSelection = (repository?: GitRepository) => {
      if (repository) {
        onRepoSelection?.(repository);
        setSelectedRepository(repository);
      } else {
        onRepoSelection?.(null); // Notify parent component that repo was cleared
        setSelectedRepository(null);
        setSelectedBranch(null);
      }
    };

    return (
      <GitRepoDropdown
        provider={selectedProvider || providers[0]}
        value={selectedRepository?.id || null}
        repositoryName={selectedRepository?.full_name || null}
        // eslint-disable-next-line i18next/no-literal-string -- example value, not translatable
        placeholder="user/repo"
        disabled={!selectedProvider || isLoadingSettings}
        onChange={handleRepoSelection}
        className="max-w-auto"
      />
    );
  };

  // Render the branch selector
  const renderBranchSelector = () => {
    const defaultBranch = selectedRepository?.main_branch || null;
    return (
      <GitBranchDropdown
        repository={selectedRepository?.full_name || null}
        provider={selectedProvider || providers[0]}
        selectedBranch={selectedBranch}
        onBranchSelect={handleBranchSelection}
        defaultBranch={defaultBranch}
        className="max-w-full"
        disabled={!selectedRepository || isLoadingSettings}
      />
    );
  };

  return (
    <div className="flex flex-col">
      {/* Skip the in-form "Open Repository" header in dialog mode — the dialog
          already shows the same title, so this would be redundant. */}
      {!onConfirm && (
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-[10px] pb-4">
            <RepoForkedIcon width={24} height={24} />
            <span className="leading-5 font-bold text-base text-white">
              {t(I18nKey.COMMON$OPEN_REPOSITORY)}
            </span>
          </div>
        </div>
      )}

      <div className="flex flex-col gap-[10px] pb-4">
        <div className="flex items-center justify-between">
          <span className="text-sm text-white font-normal leading-[22px]">
            {t(I18nKey.HOME$SELECT_OR_INSERT_URL)}
          </span>
          {renderProviderSelector()}
        </div>
        {renderRepositorySelector()}
        {renderBranchSelector()}
      </div>

      <BrandButton
        testId="repo-launch-button"
        variant="primary"
        type="button"
        isDisabled={
          !selectedRepository ||
          !selectedBranch ||
          (!onConfirm && isCreatingConversation) ||
          (providers.length > 1 && !selectedProvider) ||
          isLoadingSettings
        }
        onClick={() => {
          if (!selectedRepository || !selectedBranch) return;

          // Persist the repository to recent repositories on every confirm so
          // the home launcher and the inline path stay in sync.
          addRecentRepository(selectedRepository);

          if (onConfirm) {
            onConfirm({
              repository: selectedRepository,
              branch: selectedBranch,
              provider: selectedProvider,
            });
            return;
          }

          createConversation(
            {
              repository: {
                name: selectedRepository.full_name || "",
                gitProvider: selectedRepository.git_provider || "github",
                branch: selectedBranch.name || "main",
              },
              entryPoint: "home_repo_form",
            },
            {
              onSuccess: (data) =>
                navigate(`/conversations/${data.conversation_id}`),
            },
          );
        }}
        className="w-full"
      >
        {onConfirm
          ? t(I18nKey.BUTTON$CONFIRM)
          : !isCreatingConversation
            ? t(I18nKey.BUTTON$LAUNCH)
            : t(I18nKey.HOME$LOADING)}
      </BrandButton>
    </div>
  );
}
