let originalTitle = "";
let titleInterval: number | undefined;

const isBrowser =
  typeof window !== "undefined" && typeof document !== "undefined";

// Use a constant for the notification parameter to avoid hardcoded strings
const NOTIFICATION_PARAM = "notification";

export const browserTab = {
  startNotification(message: string) {
    if (!isBrowser) return;

    // Always capture the current title as the baseline to restore to
    originalTitle = document.title;

    // Clear any existing interval
    if (titleInterval) {
      this.stopNotification();
    }

    // Alternate between the latest baseline title and the notification message.
    // If the title changes externally (e.g., user renames conversation),
    // update the baseline so we restore to the new value when stopping.
    titleInterval = window.setInterval(() => {
      const current = document.title;
      if (current !== originalTitle && current !== message) {
        originalTitle = current;
      }
      document.title = current === message ? originalTitle : message;
    }, 1000);

    // Set favicon to indicate notification
    const favicon = document.querySelector(
      'link[rel="icon"]',
    ) as HTMLLinkElement;
    if (favicon) {
      favicon.href = favicon.href.includes(`?${NOTIFICATION_PARAM}`)
        ? favicon.href
        : `${favicon.href}?${NOTIFICATION_PARAM}`;
    }
  },

  stopNotification() {
    if (!isBrowser) return;

    if (titleInterval) {
      window.clearInterval(titleInterval);
      titleInterval = undefined;
    }
    if (originalTitle) {
      document.title = originalTitle;
    }

    // Reset favicon
    const favicon = document.querySelector(
      'link[rel="icon"]',
    ) as HTMLLinkElement;
    if (favicon) {
      favicon.href = favicon.href.replace(`?${NOTIFICATION_PARAM}`, "");
    }
  },
};
