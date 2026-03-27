/**
 * Calculate toast duration based on message length
 * @param message - The message to display
 * @param minDuration - Minimum duration in milliseconds (default: 5000 for success, 4000 for error)
 * @param maxDuration - Maximum duration in milliseconds (default: 10000)
 * @returns Duration in milliseconds
 */
export const calculateToastDuration = (
  message: string | null | undefined,
  minDuration: number = 5000,
  maxDuration: number = 10000,
): number => {
  // Handle null/undefined messages - return minDuration immediately
  if (!message) {
    return minDuration;
  }

  // Calculate duration based on reading speed (average 200 words per minute)
  // Assuming average word length of 5 characters
  const wordsPerMinute = 200;
  const charactersPerMinute = wordsPerMinute * 5;
  const charactersPerSecond = charactersPerMinute / 60;

  // Calculate time needed to read the message
  const readingTimeMs = (message.length / charactersPerSecond) * 1000;

  // Add some buffer time (50% extra) for processing
  const durationWithBuffer = readingTimeMs * 1.5;

  // Ensure duration is within min/max bounds
  return Math.min(Math.max(durationWithBuffer, minDuration), maxDuration);
};
