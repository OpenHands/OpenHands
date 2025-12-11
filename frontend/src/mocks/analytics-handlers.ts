import { http, HttpResponse } from "msw";

export const ANALYTICS_HANDLERS = [
  http.post("https://us.i.posthog.com/e", async () =>
    HttpResponse.json(null, { status: 200 }),
  ),
];
