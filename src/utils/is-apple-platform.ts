const APPLE_PLATFORM_PATTERN = /Mac|iPhone|iPad|iPod/i;

/**
 * Whether the browser runs on an Apple platform, where shortcuts use the
 * Command key (⌘) instead of Ctrl.
 */
export const isApplePlatform = (): boolean =>
  typeof navigator !== "undefined" &&
  APPLE_PLATFORM_PATTERN.test(navigator.platform);
