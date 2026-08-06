const BYTES_PER_MB = 1024 * 1024;
const DEFAULT_MAX_FILE_SIZE_MB = 25;
const DEFAULT_MAX_TOTAL_SIZE_MB = 50;
const MAX_EMBEDDED_IMAGE_SIZE_MB = 3;

interface AttachmentLimits {
  maxFileSizeMb: number;
  maxTotalSizeMb: number;
}

interface RuntimeAttachmentLimits {
  maxFileSizeMb?: unknown;
  maxTotalSizeMb?: unknown;
}

export interface FileValidationResult {
  isValid: boolean;
  errorMessage?: string;
  oversizedFiles?: string[];
}

function parsePositiveNumber(value: unknown, fallback: number): number {
  const parsed =
    typeof value === "number"
      ? value
      : typeof value === "string"
        ? Number(value.trim())
        : Number.NaN;
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function getRuntimeAttachmentLimits(): RuntimeAttachmentLimits {
  if (typeof window === "undefined") return {};

  const runtimeConfig = (window as unknown as Record<string, unknown>)[
    "__AGENT_CANVAS_ATTACHMENT_LIMITS__"
  ];
  return runtimeConfig && typeof runtimeConfig === "object"
    ? (runtimeConfig as RuntimeAttachmentLimits)
    : {};
}

export function getAttachmentLimits(): AttachmentLimits {
  const runtime = getRuntimeAttachmentLimits();
  const maxFileSizeMb = parsePositiveNumber(
    runtime.maxFileSizeMb ?? import.meta.env.VITE_MAX_ATTACHMENT_FILE_SIZE_MB,
    DEFAULT_MAX_FILE_SIZE_MB,
  );
  const configuredTotalSizeMb = parsePositiveNumber(
    runtime.maxTotalSizeMb ?? import.meta.env.VITE_MAX_ATTACHMENT_TOTAL_SIZE_MB,
    DEFAULT_MAX_TOTAL_SIZE_MB,
  );

  return {
    maxFileSizeMb,
    maxTotalSizeMb: Math.max(maxFileSizeMb, configuredTotalSizeMb),
  };
}

export function validateIndividualFileSizes(
  files: File[],
): FileValidationResult {
  const { maxFileSizeMb } = getAttachmentLimits();
  const maxFileSize = maxFileSizeMb * BYTES_PER_MB;
  const oversizedFiles = files.filter((file) => file.size > maxFileSize);

  if (oversizedFiles.length > 0) {
    const fileNames = oversizedFiles.map((file) => file.name);
    return {
      isValid: false,
      errorMessage: `Files exceeding ${maxFileSizeMb}MB are not allowed: ${fileNames.join(", ")}`,
      oversizedFiles: fileNames,
    };
  }

  return { isValid: true };
}

export function validateTotalFileSize(
  newFiles: File[],
  existingFiles: File[] = [],
): FileValidationResult {
  const { maxTotalSizeMb } = getAttachmentLimits();
  const maxTotalSize = maxTotalSizeMb * BYTES_PER_MB;
  const currentTotalSize = existingFiles.reduce(
    (sum, file) => sum + file.size,
    0,
  );
  const newFilesSize = newFiles.reduce((sum, file) => sum + file.size, 0);
  const totalSize = currentTotalSize + newFilesSize;

  if (totalSize > maxTotalSize) {
    const totalSizeMb = (totalSize / BYTES_PER_MB).toFixed(1);
    return {
      isValid: false,
      errorMessage: `Total file size would be ${totalSizeMb}MB, exceeding the ${maxTotalSizeMb}MB limit. Please select fewer or smaller files.`,
    };
  }

  return { isValid: true };
}

export function validateFiles(
  newFiles: File[],
  existingFiles: File[] = [],
): FileValidationResult {
  const individualValidation = validateIndividualFileSizes(newFiles);
  if (!individualValidation.isValid) {
    return individualValidation;
  }

  return validateTotalFileSize(newFiles, existingFiles);
}

export function validateEmbeddedImageSizes(
  images: File[],
): FileValidationResult {
  const maxEmbeddedImageSize = MAX_EMBEDDED_IMAGE_SIZE_MB * BYTES_PER_MB;
  const oversizedImages = images.filter(
    (image) => image.size > maxEmbeddedImageSize,
  );

  if (oversizedImages.length === 0) return { isValid: true };

  const fileNames = oversizedImages.map((image) => image.name);
  return {
    isValid: false,
    errorMessage: `Images embedded in chat must be ${MAX_EMBEDDED_IMAGE_SIZE_MB}MB or smaller. Upload as files instead: ${fileNames.join(", ")}`,
    oversizedFiles: fileNames,
  };
}
