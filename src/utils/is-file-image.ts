/**
 * Check if a file is an image.
 * @param file - The File object to check.
 * @returns True if the file is an image, false otherwise.
 */
export const isFileImage = (file: File): boolean =>
  file.type.startsWith("image/");
