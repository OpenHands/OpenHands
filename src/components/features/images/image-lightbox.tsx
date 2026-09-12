import React from "react";
import { useTranslation } from "react-i18next";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalCloseButton } from "#/components/shared/modals/modal-close-button";
import { I18nKey } from "#/i18n/declaration";

interface ImageLightboxProps {
  src: string;
  onClose: () => void;
}

/**
 * Full-size overlay for an image attachment. The thumbnail is a CSS downscale
 * of the same source, so no refetch is needed.
 */
export function ImageLightbox({ src, onClose }: ImageLightboxProps) {
  const { t } = useTranslation("openhands");

  // HeroUI consumes Escape on a document capture listener, so it never reaches
  // ModalBackdrop's window listener. Capture it here instead.
  React.useEffect(() => {
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.stopPropagation();
      onClose();
    };

    document.addEventListener("keydown", handleEscape, true);
    return () => document.removeEventListener("keydown", handleEscape, true);
  }, [onClose]);

  return (
    <ModalBackdrop
      onClose={onClose}
      closeOnEscape={false}
      aria-label={t(I18nKey.IMAGE$FULL_SIZE_PREVIEW)}
    >
      <div data-testid="image-lightbox" className="relative">
        <ModalCloseButton
          onClose={onClose}
          testId="image-lightbox-close"
          className="bg-black/60 text-white hover:bg-black/80"
        />
        <img
          src={src}
          alt={t(I18nKey.IMAGE$FULL_SIZE_PREVIEW)}
          className="max-h-[90vh] max-w-[90vw] rounded-lg object-contain"
        />
      </div>
    </ModalBackdrop>
  );
}
