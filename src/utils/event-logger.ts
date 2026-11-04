/**
 * A utility class for logging events. This class will only log events in development mode.
 */
class EventLogger {
  static isDevMode = process.env.NODE_ENV === "development";

  /**
   * Log an error message
   * @param error The error message
   */
  static error(error: string) {
    if (this.isDevMode) {
      console.error(error);
    }
  }
}

export default EventLogger;
