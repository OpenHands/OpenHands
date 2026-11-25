import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { ChangeFactoryModal } from "#/components/features/automations/change-factory-modal";
import { useChangeAutomationFactory } from "#/hooks/query/use-automations";
import { useFactories } from "#/hooks/query/use-factories";
import type { Automation } from "#/types/automation";
import { getApiErrorMessage } from "#/utils/api-error-message";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";

interface ChangeAutomationFactoryModalProps {
  automation: Automation;
  onClose: () => void;
  /** Called after the change was saved, with the new factory id or `null`. */
  onChanged?: (factoryId: string | null) => void;
}

/**
 * Change which factory an automation is in, with the factory list, the save,
 * and the result toasts. Mount it while the dialog is open; callers decide who
 * may open it and what follows.
 */
export function ChangeAutomationFactoryModal({
  automation,
  onClose,
  onChanged,
}: ChangeAutomationFactoryModalProps) {
  const { t } = useTranslation("openhands");
  const factoriesQuery = useFactories();
  const changeFactory = useChangeAutomationFactory();

  const handleConfirm = (nextFactoryId: string | null) => {
    changeFactory.mutate(
      { id: automation.id, factoryId: nextFactoryId },
      {
        onSuccess: () => {
          onClose();
          const nextFactory = factoriesQuery.data?.find(
            (factory) => factory.id === nextFactoryId,
          );
          displaySuccessToast(
            nextFactory
              ? t(I18nKey.AUTOMATIONS$CHANGE_FACTORY_SUCCESS, {
                  name: automation.name,
                  factory: nextFactory.name,
                })
              : t(I18nKey.AUTOMATIONS$CHANGE_FACTORY_REMOVED, {
                  name: automation.name,
                }),
          );
          onChanged?.(nextFactoryId);
        },
        onError: (error) => {
          displayErrorToast(
            getApiErrorMessage(
              error,
              t(I18nKey.AUTOMATIONS$CHANGE_FACTORY_ERROR),
            ),
          );
        },
      },
    );
  };

  return (
    <ChangeFactoryModal
      automationName={automation.name}
      isOpen
      factories={factoriesQuery.data ?? []}
      currentFactoryId={automation.factory_id}
      isLoading={factoriesQuery.isLoading}
      isPending={changeFactory.isPending}
      onConfirm={handleConfirm}
      onCancel={onClose}
    />
  );
}
