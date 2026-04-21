let originalTitle = "";
let titleInterval: number | undefined;

const isBrowser =
  typeof window !== "undefined" && typeof document !== "undefined";

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
  },
};
