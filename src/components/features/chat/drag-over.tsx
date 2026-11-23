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
            className="absolute file-drop-image"
          />
          <ArrowDownCurveIcon
            width={16}
            height={16}
            className="absolute file-drop-arrow"
          />
        </div>
        <div>
          <p>{t(I18nKey.COMMON$DROP_YOUR_FILES_HERE)}</p>
        </div>
      </div>
    </div>
  );
}
