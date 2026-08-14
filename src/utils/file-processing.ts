type ProcessedFiles = {
  successful: File[];
  failed: { file: File; error: Error }[];
};

const preserveFilesForUpload = async (
  files: File[],
): Promise<ProcessedFiles> => ({
  successful: files,
  failed: [],
});

export const processFiles = preserveFilesForUpload;
export const processImages = preserveFilesForUpload;
