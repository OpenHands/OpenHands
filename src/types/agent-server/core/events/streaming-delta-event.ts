import { BaseEvent } from "../base/event";

export interface StreamingDeltaEvent extends BaseEvent {
  kind: "StreamingDeltaEvent";
  source: "agent";
  content: string | null;
  reasoning_content: string | null;
}
