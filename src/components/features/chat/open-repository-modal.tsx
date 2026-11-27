import React, { useState, useCallback, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalBody } from "#/components/shared/modals/modal-body";
import { ModalCloseButton } from "#/components/shared/modals/modal-close-button";
import { BrandButton } from "#/components/features/settings/brand-button";
import { BaseModalTitle } from "#/components/shared/modals/confirmation-modals/base-modal";
import { I18nKey } from "#/i18n/declaration";
import { Provider } from "#/types/settings";
import { Branch, GitRepository } from "#/types/git";
import { GitRepoDropdown } from "#/components/features/home/git-repo-dropdown/git-repo-dropdown";
import { GitBranchDropdown } from "#/components/features/home/git-branch-dropdown/git-branch-dropdown";
import { GitProviderDropdown } from "#/components/features/home/git-provider-dropdown/git-provider-dropdown";
import { useGitProviderOptions } from "#/hooks/use-git-provider-options";
import {
  gitProviderOptionsMatch,
  type GitProviderOption,
} from "#/utils/git-provider-option";

interface OpenRepositoryModalProps {
  isOpen: boolean;
  onClose: () => void;
  onLaunch: (repository: GitRepository, branch: Branch) => void;
  defaultProvider?: Provider;
  /** The conversation's connected git provider, to pick among several of one type. */
  defaultGitProviderId?: number | null;
}

export function OpenRepositoryModal({
  isOpen,
  onClose,
  onLaunch,
  defaultProvider = "github",
  defaultGitProviderId = null,
}: OpenRepositoryModalProps) {
  const { t } = useTranslation("openhands");
  const { options } = useGitProviderOptions();

  const [pickedOption, setPickedOption] = useState<GitProviderOption | null>(
    null,
  );
  const selectedOption =
    options.find((option) => gitProviderOptionsMatch(option, pickedOption)) ??
    null;
  const selectedProvider = selectedOption?.provider ?? null;
  const [selectedRepository, setSelectedRepository] =
    useState<GitRepository | null>(null);
  const [selectedBranch, setSelectedBranch] = useState<Branch | null>(null);

  // Auto-select provider: single provider auto-selects, multiple uses defaultProvider if available
  useEffect(() => {
    if (selectedOption) return;
    if (options.length === 1) {
      setPickedOption(options[0]);
    } else if (options.length > 1 && defaultProvider) {
      const preferred =
        options.find(
          (option) =>
            option.provider === defaultProvider &&
            option.gitProviderId === defaultGitProviderId,
        ) ?? options.find((option) => option.provider === defaultProvider);
      if (preferred) setPickedOption(preferred);
    }
  }, [options, selectedOption, defaultProvider, defaultGitProviderId]);

  const handleProviderChange = useCallback(
    (option: GitProviderOption | null) => {
      if (gitProviderOptionsMatch(option, selectedOption)) return;
      setPickedOption(option);
      setSelectedRepository(null);
      setSelectedBranch(null);
    },
    [selectedOption],
  );

  const handleRepositoryChange = useCallback((repository?: GitRepository) => {
    if (repository) {
      setSelectedRepository(repository);
      setSelectedBranch(null);
    } else {
      setSelectedRepository(null);
      setSelectedBranch(null);
    }
  }, []);

  const handleBranchSelect = useCallback((branch: Branch | null) => {
    setSelectedBranch(branch);
  }, []);

  const handleLaunch = () => {
    if (!selectedRepository || !selectedBranch) return;

    onLaunch(selectedRepository, selectedBranch);
    setSelectedRepository(null);
    setSelectedBranch(null);
    onClose();
  };

  const handleClose = () => {
    setPickedOption(null);
    setSelectedRepository(null);
    setSelectedBranch(null);
    onClose();
  };

  if (!isOpen) return null;

  const activeProvider =
    selectedRepository?.git_provider || selectedProvider || defaultProvider;
  const activeGitProviderId =
    selectedRepository?.git_provider_id ?? selectedOption?.gitProviderId;
  const canLaunch = !!selectedRepository && !!selectedBranch;

  return (
    <ModalBackdrop onClose={handleClose}>
      <ModalBody
        width="sm"
        className="relative items-start border border-border !gap-4"
      >
        <ModalCloseButton
          onClose={handleClose}
          testId="close-open-repository-modal"
        />
        <div className="w-full pr-6">
          <BaseModalTitle title={t(I18nKey.CONVERSATION$OPEN_REPOSITORY)} />
        </div>

        <div className="flex flex-col gap-4 w-full">
          <div className="flex items-center justify-between">
            <span className="text-sm text-contrast font-normal leading-5.5">
              {t(I18nKey.CONVERSATION$SELECT_OR_INSERT_LINK)}
            </span>
            {options.length > 1 && (
              <GitProviderDropdown
                options={options}
                value={selectedOption}
                onChange={handleProviderChange}
              />
            )}
          </div>

          <div className="flex flex-col gap-2.5 w-full">
            <GitRepoDropdown
              provider={activeProvider}
              gitProviderId={activeGitProviderId}
              value={selectedRepository?.id || null}
              repositoryName={selectedRepository?.full_name || null}
              onChange={handleRepositoryChange}
              className="w-full"
            />

            <GitBranchDropdown
              repository={selectedRepository?.full_name || null}
              provider={activeProvider}
              gitProviderId={activeGitProviderId}
              selectedBranch={selectedBranch}
              onBranchSelect={handleBranchSelect}
              defaultBranch={selectedRepository?.main_branch || null}
              disabled={!selectedRepository}
              className="w-full"
            />
          </div>
        </div>

        <div
          className="flex justify-end gap-2 w-full"
          onClick={(event) => event.stopPropagation()}
        >
          <BrandButton type="button" variant="secondary" onClick={handleClose}>
            {t(I18nKey.BUTTON$CANCEL)}
          </BrandButton>
          <BrandButton
            type="button"
            variant="primary"
            onClick={handleLaunch}
            isDisabled={!canLaunch}
          >
            {t(I18nKey.BUTTON$LAUNCH)}
          </BrandButton>
        </div>
      </ModalBody>
    </ModalBackdrop>
  );
}
