import { useTranslation } from "react-i18next";
import { NavLink } from "react-router";
import { cn } from "#/utils/utils";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";
import { SettingsNavItem } from "#/constants/settings-nav";

interface SettingsNavLinkProps {
  item: SettingsNavItem;
  onClick: () => void;
}

export function SettingsNavLink({ item, onClick }: SettingsNavLinkProps) {
  const { t } = useTranslation();
  const { to, icon, text } = item;

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
      <Typography.Text className="flex h-5 w-5 shrink-0 items-center justify-center text-[#8C8C8C] group-hover:text-white transition-colors duration-200">
        {icon}
      </Typography.Text>
      <div className="min-w-0 flex-1 overflow-hidden">
        <Typography.Text
          className={cn(
            "block truncate whitespace-nowrap text-[#8C8C8C] transition-all duration-300",
            "group-hover:translate-x-1 group-hover:text-white",
          )}
        >
          {t(text as I18nKey)}
        </Typography.Text>
      </div>
    </NavLink>
  );
}
