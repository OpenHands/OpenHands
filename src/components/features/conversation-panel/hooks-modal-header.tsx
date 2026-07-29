import { useTranslation } from "react-i18next";
import { RefreshCw } from "lucide-react";
import { BaseModalTitle } from "#/components/shared/modals/confirmation-modals/base-modal";
import { ModalCloseButton } from "#/components/shared/modals/modal-close-button";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";
import { cn } from "#/utils/utils";

interface HooksModalHeaderProps {
  isLoading: boolean;
  isRefetching: boolean;
  onRefresh: () => void;
  onClose: () => void;
}

const ICON_BUTTON_CLASS =
  "rounded-md p-1 text-white hover:bg-tertiary cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed";

export function HooksModalHeader({
  isLoading,
  isRefetching,
  onRefresh,
  onClose,
}: HooksModalHeaderProps) {
  const { t } = useTranslation("openhands");
  const refreshLabel = t(I18nKey.BUTTON$REFRESH);

  return (
    <>
      <ModalCloseButton onClose={onClose} testId="close-hooks-modal" />
      <div className="flex w-full items-start justify-between gap-4 pr-10">
        <div className="flex min-w-0 flex-1 flex-col gap-2">
          <BaseModalTitle title={t(I18nKey.HOOKS_MODAL$TITLE)} />
          <Typography.Text className="text-sm text-[var(--oh-muted)]">
            {t(I18nKey.HOOKS_MODAL$WARNING)}
          </Typography.Text>
        </div>
        <StyledTooltip content={refreshLabel} placement="bottom">
          <button
            type="button"
            data-testid="refresh-hooks"
            onClick={onRefresh}
            disabled={isLoading || isRefetching}
            aria-label={refreshLabel}
            className={ICON_BUTTON_CLASS}
          >
            <RefreshCw
              size={18}
              className={cn(isRefetching && "animate-spin")}
              aria-hidden
            />
          </button>
        </StyledTooltip>
      </div>
    </>
  );
}
