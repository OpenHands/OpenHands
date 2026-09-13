import React from "react";
import { useTranslation } from "react-i18next";
import { RemoveButton } from "#/components/shared/buttons/remove-button";
import { I18nKey } from "#/i18n/declaration";
import { ImageLightbox } from "./image-lightbox";
import { Thumbnail } from "./thumbnail";

interface ImagePreviewProps {
  src: string;
  onRemove?: () => void;
  size?: "small" | "large";
}

export function ImagePreview({
  src,
  onRemove,
  size = "small",
}: ImagePreviewProps) {
  const { t } = useTranslation("openhands");
  const [isExpanded, setIsExpanded] = React.useState(false);

  return (
    <div data-testid="image-preview" className="relative w-fit shrink-0 py-1">
      <button
        type="button"
        data-testid="expand-image-button"
        aria-label={t(I18nKey.BUTTON$VIEW_FULL_SIZE_IMAGE)}
        onClick={() => setIsExpanded(true)}
        className="block cursor-zoom-in rounded-sm transition-opacity hover:opacity-80"
      >
        <Thumbnail src={src} size={size} />
      </button>
      {onRemove && (
        <RemoveButton
          onClick={onRemove}
          aria-label={t(I18nKey.BUTTON$REMOVE_IMAGE)}
          className="absolute right-[3px] top-[3px] cursor-pointer"
        />
      )}
      {isExpanded && (
        <ImageLightbox src={src} onClose={() => setIsExpanded(false)} />
      )}
    </div>
  );
}
