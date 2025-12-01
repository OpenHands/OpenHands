/**
 * Parses a date string as UTC if it doesn't have a timezone indicator.
 * This fixes the issue where ISO strings without timezone info are interpreted as local time.
 * @param dateString ISO 8601 date string
 * @returns Date object parsed as UTC
 *
 * @example
 * parseDateAsUTC("2025-12-01T11:53:37.273886"); // Parsed as UTC
 * parseDateAsUTC("2025-12-01T11:53:37.273886Z"); // Already has timezone, parsed correctly
 * parseDateAsUTC("2025-12-01T11:53:37+00:00"); // Already has timezone, parsed correctly
 */
const parseDateAsUTC = (dateString: string): Date => {
  // Check if the string already has a timezone indicator
  // Look for 'Z' (UTC), '+' (positive offset), or '-' after the time part (negative offset)
  const hasTimezone =
    dateString.includes("Z") || dateString.match(/[+-]\d{2}:\d{2}$/) !== null;

  if (hasTimezone) {
    // Already has timezone info, parse normally
    return new Date(dateString);
  }

  // No timezone indicator - append 'Z' to force UTC parsing
  return new Date(`${dateString}Z`);
};

/**
 * Formats a date into a compact string representing the time delta between the given date and the current date.
 * @param date The date to format (Date object or ISO 8601 string)
 * @returns A compact string representing the time delta between the given date and the current date
 *
 * @example
 * // now is 2024-01-01T00:00:00Z
 * formatTimeDelta(new Date("2023-12-31T23:59:59Z")); // "1s"
 * formatTimeDelta("2023-12-31T23:59:59Z"); // "1s"
 * formatTimeDelta("2025-12-01T11:53:37.273886"); // Parsed as UTC automatically
 */
export const formatTimeDelta = (date: Date | string) => {
  // Parse string dates as UTC if needed, or use Date object directly
  const dateObj = typeof date === "string" ? parseDateAsUTC(date) : date;
  const now = new Date();
  const delta = now.getTime() - dateObj.getTime();

  const seconds = Math.floor(delta / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);
  const months = Math.floor(days / 30);
  const years = Math.floor(months / 12);

  if (seconds < 60) return `${seconds}s`;
  if (minutes < 60) return `${minutes}m`;
  if (hours < 24) return `${hours}h`;
  if (days < 30) return `${days}d`;
  if (months < 12) return `${months}mo`;
  return `${years}y`;
};

/**
 * Formats a date into a MM/DD/YYYY string format.
 * @param date The date to format
 * @returns A string in MM/DD/YYYY format
 *
 * @example
 * formatDateMMDDYYYY(new Date("2025-05-30T00:15:08")); // "05/30/2025"
 * formatDateMMDDYYYY(new Date("2024-12-25T10:30:00")); // "12/25/2024"
 */
export const formatDateMMDDYYYY = (date: Date) =>
  date.toLocaleDateString("en-US", {
    month: "2-digit",
    day: "2-digit",
    year: "numeric",
  });
