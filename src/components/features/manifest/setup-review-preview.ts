export const SETUP_REVIEW_PREVIEW_QUERY_PARAM = "previewSetupReview";

/** Catalog entry whose defaults match the Review mock (RSS digest). */
export const SETUP_REVIEW_PREVIEW_ENTRY_ID = "news-digest";

/**
 * Flip off to return to the real setup flow. While this is true the Review
 * modal is forced open on every page so we can style it in place.
 */
export const FORCE_SETUP_REVIEW_PREVIEW = false;

export function isSetupReviewPreviewActive(
  search = typeof window !== "undefined" ? window.location.search : "",
): boolean {
  if (FORCE_SETUP_REVIEW_PREVIEW) return true;
  return new URLSearchParams(search).has(SETUP_REVIEW_PREVIEW_QUERY_PARAM);
}
