import { UploadedFile } from "./uploaded-file";
import { UploadedImage } from "./uploaded-image";
import { useConversationStore } from "#/stores/conversation-store";

export function UploadedFiles() {
  const {
    images,
    files,
    loadingFiles,
    loadingImages,
    imagesMarkedUploadAsFile,
    removeFile,
    removeImage,
    toggleImageUploadAsFile,
  } = useConversationStore();

  const handleRemoveFile = (index: number) => {
    removeFile(index);
  };

  const handleRemoveImage = (index: number) => {
    removeImage(index);
  };

  // Don't render anything if there are no files, images, or loading items
  if (
    images.length === 0 &&
    files.length === 0 &&
    loadingFiles.length === 0 &&
    loadingImages.length === 0
  ) {
    return null;
  }

  return (
    <div className="flex w-full flex-col gap-2 pb-4">
      <div className="flex w-full items-center gap-4 overflow-x-auto custom-scrollbar">
        {/* Regular files */}
        {files.map((file, index) => (
          <UploadedFile
            key={`file-${index}-${file.name}`}
            file={file}
            onRemove={() => handleRemoveFile(index)}
            isLoading={loadingFiles.includes(file.name)}
          />
        ))}

        {/* Loading files (files currently being processed) */}
        {loadingFiles.map((fileName, index) => {
          // Create a temporary File object for display purposes
          const tempFile = new File([], fileName);
          return (
            <UploadedFile
              key={`loading-file-${index}-${fileName}`}
              file={tempFile}
              onRemove={() => {}} // No remove action during loading
              isLoading
            />
          );
        })}

        {/* Regular images */}
        {images.map((image, index) => (
          <UploadedImage
            key={`image-${index}-${image.name}`}
            image={image}
            onRemove={() => handleRemoveImage(index)}
            isLoading={loadingImages.includes(image.name)}
            showUploadAsFileToggle
            uploadAsFileActive={imagesMarkedUploadAsFile.includes(image.name)}
            onToggleUploadAsFile={() => toggleImageUploadAsFile(image.name)}
          />
        ))}

        {/* Loading images (images currently being processed) */}
        {loadingImages.map((imageName, index) => {
          // Create a temporary File object for display purposes
          const tempImage = new File([], imageName);
          return (
            <UploadedImage
              key={`loading-image-${index}-${imageName}`}
              image={tempImage}
              onRemove={() => {}} // No remove action during loading
              isLoading
              showUploadAsFileToggle
              uploadAsFileActive={imagesMarkedUploadAsFile.includes(imageName)}
              onToggleUploadAsFile={() => toggleImageUploadAsFile(imageName)}
            />
          );
        })}
      </div>
    </div>
  );
}
