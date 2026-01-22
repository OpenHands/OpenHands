import { delay, http, HttpResponse } from "msw";

// Billing handlers for mock service worker
export const STRIPE_BILLING_HANDLERS = [
  http.get("/api/billing/credits", async () => {
    await delay();
    return HttpResponse.json({ credits: "100" });
  }),

  http.post("/api/billing/create-checkout-session", async () => {
    await delay();
    return HttpResponse.json({
      redirect_url: "https://stripe.com/some-checkout",
    });
  }),

  http.post("/api/billing/create-customer-setup-session", async () => {
    await delay();
    return HttpResponse.json({
      redirect_url: "https://stripe.com/some-customer-setup",
    });
  }),
];

// Simplified exports - subscription handlers removed as they are unused
export const STRIPE_BILLING_HANDLERS_NO_SUBSCRIPTION = STRIPE_BILLING_HANDLERS;
export const STRIPE_BILLING_HANDLERS_CANCELLED_SUBSCRIPTION =
  STRIPE_BILLING_HANDLERS;
export const STRIPE_BILLING_HANDLERS_EXPIRED_SUBSCRIPTION =
  STRIPE_BILLING_HANDLERS;
