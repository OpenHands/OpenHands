/**
 * Utility functions for chat input component
 */
/* eslint-disable no-param-reassign */

const CLIPBOARD_IMAGE_EXTENSIONS: Record<string, string> = {
  "image/png": "png",
  "image/jpeg": "jpg",
  "image/jpg": "jpg",
  "image/gif": "gif",
  "image/webp": "webp",
  "image/bmp": "bmp",
};

/**
 * Screenshots and copied images are often exposed only via
 * `clipboardData.items` (not `clipboardData.files`). Normalize unnamed
 * clipboard files so validation and loading UI have stable labels.
 */
export function normalizePastedFile(file: File): File {
  if (file.name.trim()) {
    return file;
  }

  const extension =
    CLIPBOARD_IMAGE_EXTENSIONS[file.type] ??
    (file.type.startsWith("image/") ? "png" : "bin");

  return new File([file], `pasted-image-${Date.now()}.${extension}`, {
    type: file.type,
    lastModified: file.lastModified,
  });
}

/** Matches names assigned by {@link normalizePastedFile} for clipboard screenshots. */
export const PASTED_CLIPBOARD_IMAGE_NAME = /^pasted-image-\d+\.[a-z0-9]+$/i;

export function isPastedClipboardImage(file: File): boolean {
  return PASTED_CLIPBOARD_IMAGE_NAME.test(file.name);
}

export function partitionImagesForUpload(
  images: File[],
  markedUploadAsFileNames: readonly string[],
): { imagesToEmbed: File[]; imagesAsFiles: File[] } {
  const marked = new Set(markedUploadAsFileNames);
  const imagesToEmbed: File[] = [];
  const imagesAsFiles: File[] = [];

  for (const image of images) {
    if (marked.has(image.name)) {
      imagesAsFiles.push(image);
    } else {
      imagesToEmbed.push(image);
    }
  }

  return { imagesToEmbed, imagesAsFiles };
}

/**
 * Collect files from a paste event, including clipboard image items.
 */
export function getClipboardFiles(clipboardData: DataTransfer): File[] {
  const fromFileList = Array.from(clipboardData.files);
  if (fromFileList.length > 0) {
    return fromFileList.map(normalizePastedFile);
  }

  const fromItems: File[] = [];
  for (let i = 0; i < clipboardData.items.length; i += 1) {
    const item = clipboardData.items[i];
    if (item.kind !== "file") {
      continue;
    }
    const file = item.getAsFile();
    if (file) {
      fromItems.push(normalizePastedFile(file));
    }
  }

  return fromItems;
}
/**
 * Check if contentEditable element is truly empty
 */
export const isContentEmpty = (element: HTMLDivElement | null): boolean => {
  if (!element) {
    return true;
  }
  const text = element.innerText || element.textContent || "";
  return text.trim() === "";
};

/**
 * Clear empty content from contentEditable element for placeholder display
 */
export const clearEmptyContent = (element: HTMLDivElement | null): void => {
  if (element && isContentEmpty(element)) {
    element.innerHTML = "";
    element.textContent = "";
  }
};

/**
 * Get text content from contentEditable element
 */
export const getTextContent = (element: HTMLDivElement | null): string =>
  element?.innerText || "";

/**
 * Clear text content from contentEditable element
 */
export const clearTextContent = (element: HTMLDivElement | null): void => {
  if (element) {
    element.textContent = "";
  }
};

/**
 * Clear file input value
 */
export const clearFileInput = (element: HTMLInputElement | null): void => {
  if (element) {
    element.value = "";
  }
};

/**
 * Ensure cursor stays visible when content is scrollable
 */
export const ensureCursorVisible = (element: HTMLElement | null): void => {
  if (!element) {
    return;
  }

  const selection = window.getSelection();
  if (!selection || selection.rangeCount === 0) {
    return;
  }

  const range = selection.getRangeAt(0);
  if (!range.getBoundingClientRect || !element.getBoundingClientRect) {
    return;
  }

  const rect = range.getBoundingClientRect();
  const inputRect = element.getBoundingClientRect();

  // If cursor is below the visible area, scroll to show it
  if (rect.bottom > inputRect.bottom) {
    element.scrollTop = element.scrollHeight - element.clientHeight;
  }
};

/**
 * Focus a contentEditable input and place the caret at the end of its text.
 */
export const focusContentEditableAtEnd = (
  element: HTMLElement | null,
): void => {
  if (!element) {
    return;
  }

  element.focus();

  const selection = window.getSelection();
  if (!selection) {
    return;
  }

  const range = document.createRange();
  range.selectNodeContents(element);
  range.collapse(false);
  selection.removeAllRanges();
  selection.addRange(range);
  ensureCursorVisible(element);
};
