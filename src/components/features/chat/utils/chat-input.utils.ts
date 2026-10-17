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
 * Windows IME keydown uses keyCode 229 while the composition session is
 * active. Some third-party IMEs omit `isComposing` on that keystroke.
 */
export const IME_KEY_CODE = 229;

const COMPOSING_ATTRIBUTE = "data-composing";

type ImeEventLike = {
  isComposing?: boolean;
  keyCode?: number;
  nativeEvent?: {
    isComposing?: boolean;
    keyCode?: number;
  };
};

/** DOM events, React synthetic events, and test doubles. */
export type ImeEventInput =
  | ImeEventLike
  | { nativeEvent: Event; isComposing?: boolean; keyCode?: number }
  | null
  | undefined;

/**
 * True while an IME (Chinese / Japanese / Korean) composition session is
 * active. Writing innerHTML/textContent during that window cancels the
 * session — third-party Windows IMEs are especially sensitive because they
 * often keep preedit out of the DOM until confirm.
 */
export const isImeComposingEvent = (
  event: ImeEventInput,
  composingSession = false,
): boolean => {
  if (composingSession) {
    return true;
  }
  if (!event) {
    return false;
  }
  const native =
    "nativeEvent" in event && event.nativeEvent ? event.nativeEvent : undefined;
  const nativeIme = native as
    | { isComposing?: boolean; keyCode?: number }
    | undefined;
  const isComposing =
    ("isComposing" in event && event.isComposing === true) ||
    nativeIme?.isComposing === true;
  const keyCode =
    ("keyCode" in event && typeof event.keyCode === "number"
      ? event.keyCode
      : undefined) ?? nativeIme?.keyCode;
  return isComposing || keyCode === IME_KEY_CODE;
};

export const markImeComposing = (element: HTMLElement | null): void => {
  element?.setAttribute(COMPOSING_ATTRIBUTE, "true");
};

export const clearImeComposing = (element: HTMLElement | null): void => {
  element?.removeAttribute(COMPOSING_ATTRIBUTE);
};

export const isImeComposingElement = (element: HTMLElement | null): boolean =>
  Boolean(element?.hasAttribute(COMPOSING_ATTRIBUTE));

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
 * Clear empty content from contentEditable element for placeholder display.
 * Must not run during IME composition: assigning innerHTML/textContent aborts
 * the composition session even when the DOM already looks empty.
 */
export const clearEmptyContent = (element: HTMLDivElement | null): void => {
  if (element && !isImeComposingElement(element) && isContentEmpty(element)) {
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

/**
 * Insert text at the caret (or the end when the caret is elsewhere), padded
 * with spaces so it doesn't fuse with neighbouring words.
 */
export const insertTextAtCaret = (
  element: HTMLElement | null,
  text: string,
): void => {
  if (!element) {
    return;
  }

  const selection = window.getSelection();
  if (!selection) {
    return;
  }
  if (selection.rangeCount === 0 || !element.contains(selection.anchorNode)) {
    focusContentEditableAtEnd(element);
  } else {
    element.focus();
  }

  const caret = selection.getRangeAt(0);
  const context = document.createRange();
  context.selectNodeContents(element);
  context.setEnd(caret.startContainer, caret.startOffset);
  const lead = /\S$/.test(context.toString()) ? " " : "";
  context.selectNodeContents(element);
  context.setStart(caret.endContainer, caret.endOffset);
  const trail = /^\S/.test(context.toString()) ? " " : "";

  // Same path as plain-text paste: fires `input`, keeps undo, and is a no-op
  // when the field is read-only.
  document.execCommand("insertText", false, `${lead}${text}${trail}`);
};
