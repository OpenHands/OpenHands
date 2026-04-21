import { usePostHog } from "posthog-js/react";

/**
 * Lightweight client-side analytics for UI-only events that have
 * no natural server round-trip. All server-side business events
 * go through the backend AnalyticsService instead.
 *
 * These events are tracked directly via PostHog's client SDK,
 * which automatically handles anonymous IDs and links them to
 * the user when posthog.identify() is called on login.
 */
export const useClientAnalytics = () => {
  const posthog = usePostHog();

  const trackSaasSelfhostedInquiry = ({ location }: { location: string }) => {
    posthog?.capture("saas_selfhosted_inquiry", {
      location,
    });
  };

  const trackEnterpriseLeadFormSubmitted = ({
    requestType,
    name,
    company,
    email,
    message,
  }: {
    requestType: string;
    name: string;
    company: string;
    email: string;
    message: string;
  }) => {
    posthog?.capture("enterprise_lead_form_submitted", {
      request_type: requestType,
      name,
      company,
      email,
      message,
    });
  };

  return {
    trackSaasSelfhostedInquiry,
    trackEnterpriseLeadFormSubmitted,
  };
};
