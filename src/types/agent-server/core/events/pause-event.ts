import { BaseEvent } from "../base/event";

// Pause event - indicates that agent execution was paused by user request
export interface PauseEvent extends BaseEvent {
  /**
   * The source is always "user" for pause events
   */
  source: "user";
}
