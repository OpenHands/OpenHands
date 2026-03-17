import React from "react";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

interface ModalButtonGroupProps {
  primaryText: string;
  secondaryText?: string;
  onPrimaryClick?: () => void;
  onSecondaryClick: () => void;
  isLoading?: boolean;
  primaryType?: "button" | "submit";
  primaryTestId?: string;
  secondaryTestId?: string;
  fullWidth?: boolean;
}

export function ModalButtonGroup({
  primaryText,
  secondaryText,
  onPrimaryClick,
  onSecondaryClick,
  isLoading = false,
  primaryType = "button",
  primaryTestId,
  secondaryTestId,
  fullWidth = false,
}: ModalButtonGroupProps) {
  const { t } = useTranslation();
  const closeText = secondaryText ?? t(I18nKey.BUTTON$CLOSE);

  return (
    <div className="flex gap-2 w-full">
      <BrandButton
        type={primaryType}
        variant="primary"
        onClick={onPrimaryClick}
        className={cn(
          "flex items-center justify-center",
          fullWidth ? "w-full" : "grow",
        )}
        testId={primaryTestId}
        isDisabled={isLoading}
      >
        {isLoading ? (
          <LoadingSpinner
            size="small"
            className="w-5 h-5"
            innerClassName="hidden"
            outerClassName="w-5 h-5"
          />
        ) : (
          primaryText
        )}
      </BrandButton>
      <BrandButton
        type="button"
        variant="secondary"
        onClick={onSecondaryClick}
        className={cn(fullWidth ? "w-full" : "grow")}
        testId={secondaryTestId}
        isDisabled={isLoading}
      >
        {closeText}
      </BrandButton>
    </div>
  );
}
