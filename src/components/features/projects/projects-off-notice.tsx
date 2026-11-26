import { useTranslation } from "react-i18next";
import { NavigationLink } from "#/components/shared/navigation-link";
import { I18nKey } from "#/i18n/declaration";

/** Shown on the Projects pages while the `projects` experimental feature is off. */
export function ProjectsOffNotice() {
  const { t } = useTranslation("openhands");
  return (
    <div
      data-testid="projects-off"
      className="flex flex-col gap-2 text-sm text-muted"
    >
      <p>{t(I18nKey.PROJECTS$OFF)}</p>
      <NavigationLink
        to="/settings/experimental"
        className="text-link underline w-fit"
      >
        {t(I18nKey.PROJECTS$OPEN_EXPERIMENTAL)}
      </NavigationLink>
    </div>
  );
}
