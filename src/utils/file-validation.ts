const MB = 1024 * 1024;
// Defaults mirror config/defaults.json (upload.maxFileMb / maxTotalMb).
const DEFAULT_MAX_FILE_SIZE = 3 * MB;
const DEFAULT_MAX_TOTAL_SIZE = 3 * MB;

// Runtime globals injected into index.html by the static server, which reads
// AGENT_CANVAS_MAX_FILE_MB / AGENT_CANVAS_MAX_TOTAL_MB (docker/entrypoint.sh).
// Same mechanism as __AGENT_CANVAS_BASE_PATH__ (see #/utils/base-path).
const MAX_FILE_SIZE_WINDOW_KEY = "__AGENT_CANVAS_MAX_FILE_SIZE__";
const MAX_TOTAL_SIZE_WINDOW_KEY = "__AGENT_CANVAS_MAX_TOTAL_SIZE__";

function resolveMaxSizeBytes(
  envMb: unknown,
  windowKey: string,
  fallbackBytes: number,
): number {
  // 1. Build-time env var (VITE_MAX_FILE_SIZE_MB / VITE_MAX_TOTAL_SIZE_MB).
  const envRaw = typeof envMb === "string" ? envMb.trim() : "";
  if (envRaw) {
    const envBytes = Number.parseFloat(envRaw) * MB;
    if (Number.isFinite(envBytes) && envBytes > 0) {
      return envBytes;
    }
  }
  // 2. Runtime window global injected by the static server.
  if (typeof window !== "undefined") {
    const injected = (window as unknown as Record<string, unknown>)[windowKey];
    if (
      typeof injected === "number" &&
      Number.isFinite(injected) &&
      injected > 0
    ) {
      return injected;
    }
  }
  // 3. Compiled-in default (config/defaults.json).
  return fallbackBytes;
}

export const MAX_FILE_SIZE = resolveMaxSizeBytes(
  import.meta.env.VITE_MAX_FILE_SIZE_MB,
  MAX_FILE_SIZE_WINDOW_KEY,
  DEFAULT_MAX_FILE_SIZE,
);
export const MAX_TOTAL_SIZE = resolveMaxSizeBytes(
  import.meta.env.VITE_MAX_TOTAL_SIZE_MB,
  MAX_TOTAL_SIZE_WINDOW_KEY,
  DEFAULT_MAX_TOTAL_SIZE,
);

export interface FileValidationResult {
  isValid: boolean;
  errorMessage?: string;
  oversizedFiles?: string[];
}

function formatMb(bytes: number): string {
  return (bytes / MB).toFixed(1).replace(/\.0$/, "");
}

/**
 * Validates individual file sizes
 */
export function validateIndividualFileSizes(
  files: File[],
): FileValidationResult {
  const oversizedFiles = files.filter((file) => file.size > MAX_FILE_SIZE);

  if (oversizedFiles.length > 0) {
    const fileNames = oversizedFiles.map((f) => f.name);
    return {
      isValid: false,
      errorMessage: `Files exceeding ${formatMb(MAX_FILE_SIZE)}MB are not allowed: ${fileNames.join(", ")}`,
      oversizedFiles: fileNames,
    };
  }

  return { isValid: true };
}

/**
 * Validates total file size including existing files
 */
export function validateTotalFileSize(
  newFiles: File[],
  existingFiles: File[] = [],
): FileValidationResult {
  const currentTotalSize = existingFiles.reduce(
    (sum, file) => sum + file.size,
    0,
  );
  const newFilesSize = newFiles.reduce((sum, file) => sum + file.size, 0);
  const totalSize = currentTotalSize + newFilesSize;

  if (totalSize > MAX_TOTAL_SIZE) {
    const totalSizeMB = (totalSize / MB).toFixed(1);
    return {
      isValid: false,
      errorMessage: `Total file size would be ${totalSizeMB}MB, exceeding the ${formatMb(MAX_TOTAL_SIZE)}MB limit. Please select fewer or smaller files.`,
    };
  }

  return { isValid: true };
}

/**
 * Validates both individual and total file sizes
 */
export function validateFiles(
  newFiles: File[],
  existingFiles: File[] = [],
): FileValidationResult {
  // First check individual file sizes
  const individualValidation = validateIndividualFileSizes(newFiles);
  if (!individualValidation.isValid) {
    return individualValidation;
  }

  // Then check total size
  return validateTotalFileSize(newFiles, existingFiles);
}
