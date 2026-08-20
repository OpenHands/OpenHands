import { useTranslation } from "react-i18next";
import OpenHandsLogo from "#/assets/branding/openhands-logo.svg?react";
import { NavigationLink } from "#/components/shared/navigation-link";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

const DEFAULT_LOGO_WIDTH = 46;
const DEFAULT_LOGO_HEIGHT = 30;

export type OpenHandsLogoButtonProps = {
  className?: string;
  /** Applied to the root `<svg>` (e.g. `max-w-none` so Tailwind preflight doesn’t clamp wide marks inside a narrow flex slot). */
  logoClassName?: string;
  logoWidth?: number;
  logoHeight?: number;
};

export function OpenHandsLogoButton({
  className,
  logoClassName,
  logoWidth = DEFAULT_LOGO_WIDTH,
  logoHeight = DEFAULT_LOGO_HEIGHT,
}: OpenHandsLogoButtonProps = {}) {
  const { t } = useTranslation("openhands");

  const ariaLabel = t(I18nKey.BRANDING$OPENHANDS_LOGO);

  return (
    <NavigationLink
      to="/conversations"
      aria-label={ariaLabel}
      className={cn(className)}
    >
      <OpenHandsLogo
        width={logoWidth}
        height={logoHeight}
        className={cn("shrink-0", logoClassName)}
      />
    </NavigationLink>
  );
}
