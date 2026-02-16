import { delay, http, HttpResponse } from "msw";

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
