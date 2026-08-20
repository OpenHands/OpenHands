import { useTranslation } from "react-i18next";
import { NavigationLink } from "#/components/shared/navigation-link";
import { cn } from "#/utils/utils";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";
import { SettingsNavItem } from "#/constants/settings-nav";
import { navInteractiveTransitionClassName } from "#/components/features/sidebar/sidebar-layout";

interface SettingsNavLinkProps {
  item: SettingsNavItem;
  onClick: () => void;
}

export function SettingsNavLink({ item, onClick }: SettingsNavLinkProps) {
  const { t } = useTranslation("openhands");
  const { to, icon, text } = item;

  return (
    <NavigationLink
      end
      to={to}
      onClick={onClick}
      className={({ isActive }) =>
        cn(
          "group flex items-center gap-3 p-1 sm:px-3.5 sm:py-2 rounded",
          navInteractiveTransitionClassName,
          isActive ? "bg-tertiary" : "hover:bg-[var(--oh-surface-raised)]",
          isActive ? "[&_*]:text-white" : "",
        )
      }
    >
      <Typography.Text className="flex h-5 w-5 shrink-0 items-center justify-center text-[var(--oh-muted)] group-hover:text-white">
        {icon}
      </Typography.Text>
      <div className="min-w-0 flex-1 overflow-hidden">
        <Typography.Text
          className={cn(
            "block truncate whitespace-nowrap text-[var(--oh-muted)] group-hover:text-white",
            "transition-transform duration-300 motion-reduce:transition-none group-hover:translate-x-1",
          )}
        >
          {t(text as I18nKey)}
        </Typography.Text>
      </div>
    </NavigationLink>
  );
}
