import { RemoveButton } from "#/components/shared/buttons/remove-button";
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
  return (
    <div data-testid="image-preview" className="relative w-fit shrink-0 py-1">
      <Thumbnail src={src} size={size} />
      {onRemove && (
        <RemoveButton
          onClick={onRemove}
          className="absolute right-[3px] top-[3px] cursor-pointer"
        />
      )}
    </div>
  );
}
