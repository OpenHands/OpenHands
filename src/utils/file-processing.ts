/**
 * Real file processing utilities using FileReader API
 * These functions perform actual file reading operations that take time for large files
 */

/**
 * Process a regular file by reading its content into memory
 * For large files (1GB+), this will take significant time
 */
const processFile = async (file: File): Promise<void> =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = () => {
      // File has been fully read into memory
      resolve();
    };

    reader.onerror = () => {
      reject(new Error(`Failed to read file: ${file.name}`));
    };

    reader.onabort = () => {
      reject(new Error(`File reading was aborted: ${file.name}`));
    };

    // Read the file as ArrayBuffer - this takes time for large files
    // For a 1GB file, this can take several seconds
    reader.readAsArrayBuffer(file);
  });

/**
 * Process an image file by reading its content
 * This validates the image can be read and prepares it for display
 */
const processImage = async (image: File): Promise<void> =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = () => {
      // Image has been read and is ready for display
      resolve();
    };

    reader.onerror = () => {
      reject(new Error(`Failed to read image: ${image.name}`));
    };

    reader.onabort = () => {
      reject(new Error(`Image reading was aborted: ${image.name}`));
    };

    // Read the image as data URL - this takes time for large images
    reader.readAsDataURL(image);
  });

/**
 * Process multiple files concurrently with individual error handling
 */
export const processFiles = async (
  files: File[],
): Promise<{ successful: File[]; failed: { file: File; error: Error }[] }> => {
  const results = await Promise.allSettled(
    files.map(async (file) => {
      await processFile(file);
      return file;
    }),
  );

  const successful: File[] = [];
  const failed: { file: File; error: Error }[] = [];

  results.forEach((result, index) => {
    if (result.status === "fulfilled") {
      successful.push(result.value);
    } else {
      failed.push({ file: files[index], error: result.reason });
    }
  });

  return { successful, failed };
};

/**
 * Process multiple images concurrently with individual error handling
 */
export const processImages = async (
  images: File[],
): Promise<{ successful: File[]; failed: { file: File; error: Error }[] }> => {
  const results = await Promise.allSettled(
    images.map(async (image) => {
      await processImage(image);
      return image;
    }),
  );

  const successful: File[] = [];
  const failed: { file: File; error: Error }[] = [];

  results.forEach((result, index) => {
    if (result.status === "fulfilled") {
      successful.push(result.value);
    } else {
      failed.push({ file: images[index], error: result.reason });
    }
  });

  return { successful, failed };
};
