import { useMutation } from "@tanstack/react-query";
import { analyticsEventsService } from "#/api/analytics-service/analytics-events.api";
import { Provider } from "#/types/settings";

/**
 * Mutation hook that notifies the server when the user clicks the
 * "Pull Request" button. Posts to the generic
 * `POST /api/analytics/events` endpoint with `event_type =
 * "create pr button clicked"`; the server fires the matching PostHog event.
 *
 * Tracking is fire-and-forget: errors are swallowed so a telemetry outage
 * never blocks the user's primary action of submitting the prompt.
 */
export const useTrackCreatePrButtonClicked = () =>
  useMutation({
    mutationFn: (gitProvider: Provider | null) =>
      analyticsEventsService.trackEvent({
        event_type: "create pr button clicked",
        git_provider: gitProvider,
      }),
    // Intentionally swallow errors - analytics must not block the UX.
    onError: () => {},
  });
