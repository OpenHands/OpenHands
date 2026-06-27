import { openHands } from "../open-hands-axios";
import { Provider } from "#/types/settings";

export type AnalyticsEventResponse = {
  status: string;
};

/**
 * Tagged union of every analytics event the frontend can fire server-side via
 * `POST /api/analytics/events`. Each member's `event_type` is the exact
 * PostHog event name to capture; the remaining fields become the event's
 * properties. Mirrors the Pydantic discriminated union in
 * `enterprise/server/routes/analytics_events.py`.
 *
 * To add a new event:
 *  1. Add another member to this union with a `event_type` literal and
 *     typed properties.
 *  2. Add the matching Pydantic model on the backend.
 */
export type FrontendAnalyticsEvent = {
  event_type: "create pr button clicked";
  git_provider: Provider | null;
};

export const analyticsEventsService = {
  /**
   * Fire a frontend-originated PostHog event server-side.
   *
   * Analytics moved server-side in #14006, so the browser cannot capture
   * directly any more — it must hand the event to this endpoint, which
   * resolves the authenticated user and forwards the event to PostHog with
   * the right distinct_id / org group context.
   */
  trackEvent: async (
    event: FrontendAnalyticsEvent,
  ): Promise<AnalyticsEventResponse> => {
    const { data } = await openHands.post<AnalyticsEventResponse>(
      "/api/analytics/events",
      event,
    );
    return data;
  },
};
