import React from "react";
import { useTranslation } from "react-i18next";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import { I18nKey } from "#/i18n/declaration";

interface ArchivedDisabledTooltipProps {
  isDisabled: boolean;
  children: React.ReactNode;
  placement?: "top" | "bottom" | "left" | "right";
}

export function ArchivedDisabledTooltip({
  isDisabled,
  children,
  placement = "right",
}: ArchivedDisabledTooltipProps) {
  const { t } = useTranslation("openhands");

  if (!isDisabled) {
    return children;
  }

  return (
    <StyledTooltip
      content={t(I18nKey.CONVERSATION$UNAVAILABLE_FOR_ARCHIVES)}
      placement={placement}
      tooltipClassName="bg-white text-black text-xs font-medium leading-5"
    >
      <span className="block w-full">{children}</span>
    </StyledTooltip>
  );
}
