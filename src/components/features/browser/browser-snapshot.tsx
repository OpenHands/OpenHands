import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";

interface BrowserSnaphsotProps {
  src: string;
}

export function BrowserSnapshot({ src }: BrowserSnaphsotProps) {
  const { t } = useTranslation("openhands");

  return (
    <img
      src={src}
      className="block w-full h-auto"
      alt={t(I18nKey.BROWSER$SCREENSHOT_ALT)}
    />
  );
}
