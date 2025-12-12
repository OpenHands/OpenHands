import { useState } from "react";
import { useTranslation } from "react-i18next";
import { FaCircleInfo } from "react-icons/fa6";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalBody } from "#/components/shared/modals/modal-body";
import { BrandButton } from "../settings/brand-button";
import { I18nKey } from "#/i18n/declaration";
import { useMicroagentManagementStore } from "#/state/microagent-management-store";
import XIcon from "#/icons/x.svg?react";
import { cn, getRepoMdCreatePrompt } from "#/utils/utils";
import { LearnThisRepoFormData } from "#/types/microagent-management";

interface MicroagentManagementLearnThisRepoModalProps {
  onConfirm: (formData: LearnThisRepoFormData) => void;
  onCancel: () => void;
  isLoading: boolean;
}

export function MicroagentManagementLearnThisRepoModal({
  onConfirm,
  onCancel,
  isLoading = false,
}: MicroagentManagementLearnThisRepoModalProps) {
  const { t } = useTranslation();

  const [query, setQuery] = useState<string>("");

  const { selectedRepository } = useMicroagentManagementStore();

  const onSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    const finalQuery = getRepoMdCreatePrompt(
      selectedRepository?.git_provider || "github",
      query.trim(),
    );

    onConfirm({
      query: finalQuery,
    });
  };

  const handleConfirm = () => {
    const finalQuery = getRepoMdCreatePrompt(
      selectedRepository?.git_provider || "github",
      query.trim(),
    );

    onConfirm({
      query: finalQuery,
    });
  };

  return (
    <ModalBackdrop onClose={onCancel}>
      <ModalBody
        className="items-start rounded-[12px] p-6 min-w-[611px]"
        data-testid="learn-this-repo-modal"
      >
        <div className="flex flex-col gap-2 w-full">
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-2">
              <h2
                className="text-white text-xl font-medium"
                data-testid="modal-title"
              >
                {t(I18nKey.MICROAGENT_MANAGEMENT$LEARN_THIS_REPO_MODAL_TITLE)}
              </h2>
              <a
                href="https://docs.all-hands.dev/usage/prompting/microagents-overview#microagents-overview"
                target="_blank"
                rel="noopener noreferrer"
                data-testid="modal-info-link"
              >
                <FaCircleInfo className="text-primary" />
              </a>
            </div>
            <button
              type="button"
              onClick={onCancel}
              className="cursor-pointer"
              data-testid="modal-close-button"
            >
              <XIcon width={24} height={24} color="#F9FBFE" />
            </button>
          </div>
          <span
            className="text-white text-sm font-normal"
            data-testid="modal-description"
          >
            {t(I18nKey.MICROAGENT_MANAGEMENT$LEARN_THIS_REPO_MODAL_DESCRIPTION)}
          </span>
        </div>
        <form
          data-testid="learn-this-repo-form"
          onSubmit={onSubmit}
          className="flex flex-col gap-6 w-full"
        >
          <label
            htmlFor="query-input"
            className="flex flex-col gap-2 w-full text-sm font-normal"
          >
            {t(
              I18nKey.MICROAGENT_MANAGEMENT$WHAT_YOU_WOULD_LIKE_TO_KNOW_ABOUT_THIS_REPO,
            )}
            <textarea
              required
              data-testid="query-input"
              name="query-input"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={t(
                I18nKey.MICROAGENT_MANAGEMENT$DESCRIBE_WHAT_TO_KNOW_ABOUT_THIS_REPO,
              )}
              rows={6}
              className={cn(
                "bg-tertiary border border-[#717888] bg-[#454545] w-full rounded-sm p-2 placeholder:italic placeholder:text-tertiary-alt resize-none",
                "disabled:bg-[#2D2F36] disabled:border-[#2D2F36] disabled:cursor-not-allowed",
              )}
            />
          </label>
        </form>
        <div
          className="flex items-center justify-end gap-2 w-full"
          onClick={(event) => event.stopPropagation()}
          data-testid="modal-actions"
        >
          <BrandButton
            type="button"
            variant="secondary"
            onClick={onCancel}
            testId="cancel-button"
          >
            {t(I18nKey.BUTTON$CANCEL)}
          </BrandButton>
          <BrandButton
            type="button"
            variant="primary"
            onClick={handleConfirm}
            testId="confirm-button"
            isDisabled={isLoading}
          >
            {isLoading ? t(I18nKey.HOME$LOADING) : t(I18nKey.MICROAGENT$LAUNCH)}
          </BrandButton>
        </div>
      </ModalBody>
    </ModalBackdrop>
  );
}
