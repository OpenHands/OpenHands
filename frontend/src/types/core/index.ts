import { OpenHandsAction } from "./actions";
import { OpenHandsObservation } from "./observations";
import { OpenHandsVariance } from "./variances";

/**
 * @deprecated Will be removed once we fully transition to v1 events
 */
export type OpenHandsParsedEvent =
  | OpenHandsAction
  | OpenHandsObservation
  | OpenHandsVariance;
