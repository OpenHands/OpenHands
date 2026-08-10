import { useTranslation } from "react-i18next";
import { NavigationLink } from "#/components/shared/navigation-link";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

const DEFAULT_LOGO_WIDTH = 46;
const DEFAULT_LOGO_HEIGHT = 30;

export type CortexLogoButtonProps = {
  className?: string;
  logoClassName?: string;
  logoWidth?: number;
  logoHeight?: number;
};

export function CortexLogoButton({
  className,
  logoClassName,
  logoWidth = DEFAULT_LOGO_WIDTH,
  logoHeight = DEFAULT_LOGO_HEIGHT,
}: CortexLogoButtonProps = {}) {
  const { t } = useTranslation("openhands");
  const brandName = t(I18nKey.BRANDING$OPENHANDS);

  return (
    <NavigationLink
      to="/conversations"
      aria-label={brandName}
      className={cn("flex items-center gap-2", className)}
    >
      {/* Sleek, premium, futuristic geometric brain/neural node logo */}
      <svg
        width={logoWidth}
        height={logoHeight}
        viewBox="0 0 100 100"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className={cn("shrink-0", logoClassName)}
      >
        <circle
          cx="50"
          cy="50"
          r="45"
          stroke="currentColor"
          strokeWidth="2"
          strokeDasharray="4 4"
          className="text-indigo-500/30"
        />
        {/* Connection lines */}
        <path
          d="M50 20 L30 50 L50 80 L70 50 Z"
          stroke="currentColor"
          strokeWidth="2"
          className="text-indigo-400"
          strokeLinejoin="round"
        />
        <path
          d="M30 50 L70 50"
          stroke="currentColor"
          strokeWidth="1.5"
          className="text-emerald-400"
          strokeDasharray="2 2"
        />
        <path
          d="M50 20 L50 80"
          stroke="currentColor"
          strokeWidth="1.5"
          className="text-emerald-400"
          strokeDasharray="2 2"
        />
        {/* Core nodes */}
        <circle cx="50" cy="20" r="6" fill="#10B981" /> {/* Emerald Node */}
        <circle cx="30" cy="50" r="6" fill="#6366F1" /> {/* Indigo Node */}
        <circle cx="70" cy="50" r="6" fill="#6366F1" /> {/* Indigo Node */}
        <circle cx="50" cy="80" r="6" fill="#10B981" /> {/* Emerald Node */}
        <circle
          cx="50"
          cy="50"
          r="10"
          fill="currentColor"
          className="text-indigo-600 animate-pulse"
        />
        <circle cx="50" cy="50" r="4" fill="#FFFFFF" />
      </svg>
      <span className="font-sans font-bold text-lg tracking-wider text-white hidden md:inline-block">
        {brandName.toUpperCase()}
      </span>
    </NavigationLink>
  );
}
