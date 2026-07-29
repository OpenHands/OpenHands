/**
 * A utility class for logging events. This class will only log events in development mode.
 */
class EventLogger {
  static isDevMode = process.env.NODE_ENV === "development";

  /**
   * Format and log a message event
   * @param event The raw event object
   */
  static message(event: MessageEvent) {
    if (this.isDevMode) {
      console.warn(JSON.stringify(JSON.parse(event.data.toString()), null, 2));
    }
  }

  /**
   * Log an event with a name
   * @param event The raw event object
   * @param name The name of the event
   */
  static event(event: Event, name?: string) {
    if (this.isDevMode) {
      console.warn(name || "EVENT", event);
    }
  }

  /**
   * Log a warning message
   * @param warning The warning message
   */
  static warning(warning: string) {
    if (this.isDevMode) {
      console.warn(warning);
    }
  }

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
