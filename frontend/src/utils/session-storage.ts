// CTA locations that can be dismissed
export type CTALocation = "homepage";

// Generate storage key for a CTA location
const getCTAKey = (location: CTALocation): string =>
  `${location}-cta-dismissed`;

/**
 * Set a CTA as dismissed in local storage (persists across tabs)
 * @param location The CTA location to dismiss
 */
export const setCTADismissed = (location: CTALocation): void => {
  localStorage.setItem(getCTAKey(location), "true");
};

/**
 * Check if a CTA has been dismissed
 * @param location The CTA location to check
 * @returns true if dismissed, false otherwise
 */
export const isCTADismissed = (location: CTALocation): boolean =>
  localStorage.getItem(getCTAKey(location)) === "true";
