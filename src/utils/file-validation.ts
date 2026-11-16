import {
  DEFAULT_ATTACHMENT_LIMIT_MB,
  getProviderImageLimitBytes,
  getProviderImageLimitMb,
  isImageAttachment,
} from "#/utils/provider-upload-limits";

export interface FileValidationResult {
  isValid: boolean;
  errorMessage?: string;
  oversizedFiles?: string[];
}

/**
 * The ceilings applied to attachments.
 *
 * `imageLimitBytes` is the provider's documented limit for an image and
 * `attachmentLimitBytes` covers every other file type, which the provider
 * never receives inline. Both are exposed in megabytes so an error message can
 * name the number the user is held to instead of a hard-coded constant.
 */
export interface UploadLimits {
  imageLimitBytes: number;
  imageLimitMb: number;
  attachmentLimitBytes: number;
  attachmentLimitMb: number;
  /** The provider the image limit came from, or null for the default limit. */
  provider: string | null;
}

const ATTACHMENT_LIMIT_BYTES = DEFAULT_ATTACHMENT_LIMIT_MB * 1024 * 1024;

/**
 * Resolve the upload limits for a provider.
 *
 * A provider with no documented image limit falls back to the default limit,
 * so an incomplete provider table can only make the limit stricter than the
 * provider allows, never more permissive.
 */
export function resolveUploadLimits(provider?: string | null): UploadLimits {
  return {
    imageLimitBytes: getProviderImageLimitBytes(provider),
    imageLimitMb: getProviderImageLimitMb(provider),
    attachmentLimitBytes: ATTACHMENT_LIMIT_BYTES,
    attachmentLimitMb: DEFAULT_ATTACHMENT_LIMIT_MB,
    provider: provider ?? null,
  };
}

/**
 * The limit that applies to a single file, given its type.
 */
function limitForFile(
  file: File,
  limits: UploadLimits,
): {
  limitBytes: number;
  limitMb: number;
} {
  return isImageAttachment(file)
    ? { limitBytes: limits.imageLimitBytes, limitMb: limits.imageLimitMb }
    : {
        limitBytes: limits.attachmentLimitBytes,
        limitMb: limits.attachmentLimitMb,
      };
}

/**
 * Validate individual file sizes against the limit that applies to each file.
 *
 * An image is held to the provider's limit and any other file to the generic
 * attachment limit, so the reported ceiling is the one the file was actually
 * measured against.
 */
export function validateIndividualFileSizes(
  files: File[],
  limits: UploadLimits = resolveUploadLimits(),
): FileValidationResult {
  const oversized = files
    .map((file) => ({ file, limit: limitForFile(file, limits) }))
    .filter(({ file, limit }) => file.size > limit.limitBytes);

  if (oversized.length > 0) {
    const fileNames = oversized.map(({ file }) => file.name);
    // One message per limit, so the user sees both the images that were too
    // large for the provider and the files that were too large outright.
    const byLimit = new Map<number, string[]>();
    for (const { file, limit } of oversized) {
      const names = byLimit.get(limit.limitMb) ?? [];
      names.push(file.name);
      byLimit.set(limit.limitMb, names);
    }
    const message = [...byLimit.entries()]
      .map(
        ([limitMb, names]) =>
          `Files exceeding ${limitMb}MB are not allowed: ${names.join(", ")}`,
      )
      .join(" ");

    return {
      isValid: false,
      errorMessage: message,
      oversizedFiles: fileNames,
    };
  }

  return { isValid: true };
}

/**
 * Validate the combined size of the new and existing files.
 */
export function validateTotalFileSize(
  newFiles: File[],
  existingFiles: File[] = [],
  limitBytes: number = ATTACHMENT_LIMIT_BYTES,
  limitMb: number = DEFAULT_ATTACHMENT_LIMIT_MB,
): FileValidationResult {
  const currentTotalSize = existingFiles.reduce(
    (sum, file) => sum + file.size,
    0,
  );
  const newFilesSize = newFiles.reduce((sum, file) => sum + file.size, 0);
  const totalSize = currentTotalSize + newFilesSize;

  if (totalSize > limitBytes) {
    const totalSizeMB = (totalSize / (1024 * 1024)).toFixed(1);
    return {
      isValid: false,
      errorMessage: `Total file size would be ${totalSizeMB}MB, exceeding the ${limitMb}MB limit. Please select fewer or smaller files.`,
    };
  }

  return { isValid: true };
}

/**
 * Validate both individual and total file sizes against the limits.
 */
export function validateFiles(
  newFiles: File[],
  existingFiles: File[] = [],
  limits: UploadLimits = resolveUploadLimits(),
): FileValidationResult {
  // First check individual file sizes
  const individualValidation = validateIndividualFileSizes(newFiles, limits);
  if (!individualValidation.isValid) {
    return individualValidation;
  }

  // Then check total size
  return validateTotalFileSize(
    newFiles,
    existingFiles,
    limits.attachmentLimitBytes,
    limits.attachmentLimitMb,
  );
}
