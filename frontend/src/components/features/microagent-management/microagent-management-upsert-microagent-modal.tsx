import { useEffect, useState, useMemo } from "react";
import { useTranslation } from "react-i18next";
import { FaCircleInfo } from "react-icons/fa6";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalBody } from "#/components/shared/modals/modal-body";
import { BrandButton } from "../settings/brand-button";
import { I18nKey } from "#/i18n/declaration";
import { useMicroagentManagementStore } from "#/state/microagent-management-store";
import XIcon from "#/icons/x.svg?react";
import { cn, extractRepositoryInfo } from "#/utils/utils";
import { BadgeInput } from "#/components/shared/inputs/badge-input";
import { MicroagentFormData } from "#/types/microagent-management";
import { GitRepository } from "#/types/git";
import { useRepositoryMicroagentContent } from "#/hooks/query/use-repository-microagent-content";

interface MicroagentManagementUpsertMicroagentModalProps {
  onConfirm: (formData: MicroagentFormData) => void;
  onCancel: () => void;
  isLoading: boolean;
  isUpdate?: boolean;
}

export function MicroagentManagementUpsertMicroagentModal({
  onConfirm,
  onCancel,
  isLoading = false,
  isUpdate = false,
}: MicroagentManagementUpsertMicroagentModalProps) {
  const { t } = useTranslation();

  const [triggers, setTriggers] = useState<string[]>([]);
  const [query, setQuery] = useState<string>("");

  const { selectedRepository, selectedMicroagentItem } =
    useMicroagentManagementStore();

  const { microagent } = selectedMicroagentItem ?? {};

  // Extract owner and repo from full_name for content API
  const { owner, repo, filePath } = extractRepositoryInfo(
    selectedRepository,
    microagent,
  );

  // Fetch microagent content when updating
  const { data: microagentContentData, isLoading: isLoadingContent } =
    useRepositoryMicroagentContent(owner, repo, filePath, true);

  // Populate form fields with existing microagent data when updating
  useEffect(() => {
    if (isUpdate && microagentContentData) {
      setQuery(microagentContentData.content);
      setTriggers(microagentContentData.triggers || []);
    }
  }, [isUpdate, microagentContentData]);

  const modalTitle = useMemo(() => {
    if (isUpdate) {
      return t(I18nKey.MICROAGENT_MANAGEMENT$UPDATE_MICROAGENT);
    }

    if (selectedRepository) {
      return `${t(I18nKey.MICROAGENT_MANAGEMENT$ADD_A_MICROAGENT_TO)} ${(selectedRepository as GitRepository).full_name}`;
    }

    return t(I18nKey.MICROAGENT_MANAGEMENT$ADD_A_MICROAGENT);
  }, [isUpdate, selectedRepository, t]);

  const modalDescription = useMemo(() => {
    if (isUpdate) {
      return t(
        I18nKey.MICROAGENT_MANAGEMENT$UPDATE_MICROAGENT_MODAL_DESCRIPTION,
      );
    }

    return t(I18nKey.MICROAGENT_MANAGEMENT$ADD_MICROAGENT_MODAL_DESCRIPTION);
  }, [isUpdate, t]);

  const onSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (!query.trim()) {
      return;
    }

    onConfirm({
      query: query.trim(),
      triggers,
      microagentPath: microagent?.path || "",
    });
  };

  const handleConfirm = () => {
    if (!query.trim()) {
      return;
    }

    onConfirm({
      query: query.trim(),
      triggers,
      microagentPath: microagent?.path || "",
    });
  };

  return (
    <ModalBackdrop onClose={onCancel}>
      <ModalBody className="items-start rounded-[12px] p-6 min-w-[611px]">
        <div className="flex flex-col gap-2 w-full">
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-2">
              <h2 className="text-white text-xl font-medium">{modalTitle}</h2>
              <a
                href="https://docs.all-hands.dev/usage/prompting/microagents-overview#microagents-overview"
                target="_blank"
                rel="noopener noreferrer"
              >
                <FaCircleInfo className="text-primary" />
              </a>
            </div>
            <button type="button" onClick={onCancel} className="cursor-pointer">
              <XIcon width={24} height={24} color="#F9FBFE" />
            </button>
          </div>
          <span className="text-white text-sm font-normal">
            {modalDescription}
          </span>
        </div>
        <form
          data-testid="add-microagent-modal"
          onSubmit={onSubmit}
          className="flex flex-col gap-6 w-full"
        >
          <label
            htmlFor="query-input"
            className="flex flex-col gap-2 w-full text-sm font-normal"
          >
            {t(I18nKey.MICROAGENT_MANAGEMENT$WHAT_TO_DO)}
            <textarea
              required
              data-testid="query-input"
              name="query-input"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={t(I18nKey.MICROAGENT_MANAGEMENT$DESCRIBE_WHAT_TO_DO)}
              rows={6}
              className={cn(
                "bg-tertiary border border-[#717888] bg-[#454545] w-full rounded-sm p-2 placeholder:italic placeholder:text-tertiary-alt resize-none",
                "disabled:bg-[#2D2F36] disabled:border-[#2D2F36] disabled:cursor-not-allowed",
              )}
            />
          </label>
          <label
            htmlFor="trigger-input"
            className="flex flex-col gap-2.5 w-full text-sm"
          >
            <div className="flex items-center gap-2">
              {t(I18nKey.MICROAGENT_MANAGEMENT$ADD_TRIGGERS)}
              <a
                href="https://docs.all-hands.dev/usage/prompting/microagents-keyword"
                target="_blank"
                rel="noopener noreferrer"
              >
                <FaCircleInfo className="text-primary" />
              </a>
            </div>
            <BadgeInput
              name="trigger-input"
              value={triggers}
              placeholder={t("MICROAGENT$TYPE_TRIGGER_SPACE")}
              onChange={setTriggers}
            />
            <span className="text-xs text-[#ffffff80] font-normal">
              {t(
                I18nKey.MICROAGENT_MANAGEMENT$HELP_TEXT_DESCRIBING_VALID_TRIGGERS,
              )}
            </span>
          </label>
        </form>
        <div
          className="flex items-center justify-end gap-2 w-full"
          onClick={(event) => event.stopPropagation()}
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
            isDisabled={
              !query.trim() || isLoading || (isUpdate && isLoadingContent) // Disable while loading content for updates
            }
          >
            {isLoading || (isUpdate && isLoadingContent)
              ? t(I18nKey.HOME$LOADING)
              : t(I18nKey.MICROAGENT$LAUNCH)}
          </BrandButton>
        </div>
      </ModalBody>
    </ModalBackdrop>
  );
}
