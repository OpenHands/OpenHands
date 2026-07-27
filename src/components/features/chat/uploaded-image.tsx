import React from "react";
import { LoaderCircle } from "lucide-react";
import { PastedImageUploadAsFileButton } from "./pasted-image-upload-as-file-button";
import { RemoveFileButton } from "./remove-file-button";

interface UploadedImageProps {
  image: File;
  onRemove: () => void;
  isLoading?: boolean;
  showUploadAsFileToggle?: boolean;
  uploadAsFileActive?: boolean;
  onToggleUploadAsFile?: () => void;
}

export function UploadedImage({
  image,
  onRemove,
  isLoading = false,
  showUploadAsFileToggle = false,
  uploadAsFileActive = false,
  onToggleUploadAsFile,
}: UploadedImageProps) {
  const [imageUrl, setImageUrl] = React.useState<string>("");

  React.useEffect(() => {
    // Create object URL for image preview
    const url = URL.createObjectURL(image);
    setImageUrl(url);

    // Cleanup function to revoke object URL
    return () => {
      URL.revokeObjectURL(url);
    };
  }, [image]);

  return (
    <div className="group relative flex h-[49px] w-[51px] min-h-[49px] min-w-[51px] items-center justify-center rounded-lg bg-[var(--oh-interactive-hover)]">
      {isLoading ? (
        <LoaderCircle className="animate-spin w-5 h-5" color="white" />
      ) : (
        imageUrl && (
          <img
            src={imageUrl}
            alt={image.name}
            className="h-full w-full rounded-lg object-cover"
          />
        )
      )}
      <RemoveFileButton onClick={onRemove} />
      {showUploadAsFileToggle && onToggleUploadAsFile && (
        <PastedImageUploadAsFileButton
          active={uploadAsFileActive}
          onToggle={onToggleUploadAsFile}
        />
      )}
    </div>
  );
}
