import React from "react";
import { LoaderCircle } from "lucide-react";
import { RemoveFileButton } from "./remove-file-button";

interface UploadedImageProps {
  image: File;
  onRemove: () => void;
  isLoading?: boolean;
}

export function UploadedImage({
  image,
  onRemove,
  isLoading = false,
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
    <div className="group min-w-[51px] min-h-[49px] w-[51px] h-[49px] rounded-lg bg-[#525252] relative flex items-center justify-center">
      <RemoveFileButton onClick={onRemove} />
      {isLoading ? (
        <LoaderCircle className="animate-spin w-5 h-5" color="white" />
      ) : (
        imageUrl && (
          <img
            src={imageUrl}
            alt={image.name}
            className="w-full h-full object-cover rounded-lg"
          />
        )
      )}
    </div>
  );
}
