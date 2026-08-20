import { useTranslation } from "react-i18next";
import ImageIcon from "#/icons/image.svg?react";
import ArrowDownCurveIcon from "#/icons/arrow-down-curve.svg?react";
import { I18nKey } from "#/i18n/declaration";

export function DragOver() {
  const { t } = useTranslation("openhands");

  return (
    <div className="drag-over">
      <div className="drag-over-content-wrapper">
        <div className="relative">
          <ImageIcon
            width={36}
            height={36}
            className="rotate-[-27deg] absolute top-[-40px] left-[-10px]"
          />
          <ArrowDownCurveIcon
            width={16}
            height={16}
            className="absolute top-[-16px] left-[-20px]"
          />
        </div>
        <div className="drag-over-content">
          <p>{t(I18nKey.COMMON$DROP_YOUR_FILES_HERE)}</p>
        </div>
      </div>
    </div>
  );
}
