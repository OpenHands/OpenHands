import { useTranslation } from "react-i18next";
import { NavLink } from "react-router";
import { cn } from "#/utils/utils";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";
import { SettingsNavItem } from "#/constants/settings-nav";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";

interface SettingsNavLinkProps {
  item: SettingsNavItem;
  onClick: () => void;
  disabled?: boolean;
  disabledReason?: string;
}

export function SettingsNavLink({
  item,
  onClick,
  disabled,
  disabledReason,
}: SettingsNavLinkProps) {
  const { t } = useTranslation();
  const { to, icon, text } = item;

  const linkContent = (isActive: boolean) => (
    <>
      <Typography.Text
        className={cn(
          "flex h-5 w-5 shrink-0 items-center justify-center transition-colors duration-200",
          disabled ? "text-[#4C4C4C]" : "text-[#8C8C8C] group-hover:text-white",
          isActive && !disabled ? "text-white" : "",
        )}
      >
        {icon}
      </Typography.Text>
      <div className="min-w-0 flex-1 overflow-hidden">
        <Typography.Text
          className={cn(
            "block truncate whitespace-nowrap transition-all duration-300",
            disabled
              ? "text-[#4C4C4C]"
              : "text-[#8C8C8C] group-hover:translate-x-1 group-hover:text-white",
            isActive && !disabled ? "text-white" : "",
          )}
        >
          {t(text as I18nKey)}
        </Typography.Text>
      </div>
    </>
  );

  if (disabled) {
    const tooltipContent = disabledReason
      ? t(I18nKey.SETTINGS$AGENT_DISABLED_TOOLTIP, {
          agentName: disabledReason,
        })
      : undefined;

    const inner = (
      <div
        className="flex items-center gap-3 p-1 sm:px-3.5 sm:py-2 rounded cursor-not-allowed opacity-50"
        aria-disabled="true"
      >
        {linkContent(false)}
      </div>
    );

    return tooltipContent ? (
      <StyledTooltip content={tooltipContent} placement="right">
        {inner}
      </StyledTooltip>
    ) : (
      inner
    );
  }

  return (
    <NavLink
      end
      to={to}
      onClick={onClick}
      className={({ isActive }) =>
        cn(
          "group flex items-center gap-3 p-1 sm:px-3.5 sm:py-2 rounded transition-all duration-200",
          isActive ? "bg-[#1f1f1f99]" : "hover:bg-[#1f1f1f99]",
          isActive ? "[&_*]:text-white" : "",
        )
      }
    >
      {({ isActive }) => linkContent(isActive)}
    </NavLink>
  );
}
