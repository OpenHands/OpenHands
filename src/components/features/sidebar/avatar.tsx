import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";

interface AvatarProps {
  src: string;
}

export function Avatar({ src }: AvatarProps) {
  const { t } = useTranslation("openhands");
  return (
    <img
      src={src}
      alt={t(I18nKey.AVATAR$ALT_TEXT)}
      className="w-full h-full rounded-full"
    />
  );
}
