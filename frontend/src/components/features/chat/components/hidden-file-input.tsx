import React from "react";

interface HiddenFileInputProps {
  fileInputRef: React.RefObject<HTMLInputElement | null>;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  /** MIME types or extensions to accept, e.g. "*/*" or ".pdf,.doc" */
  accept?: string;
}

export function HiddenFileInput({
  fileInputRef,
  onChange,
  accept = "*/*",
}: HiddenFileInputProps) {
  return (
    <input
      type="file"
      ref={fileInputRef}
      multiple
      accept={accept}
      style={{ display: "none" }}
      onChange={onChange}
      data-testid="upload-image-input"
    />
  );
}
