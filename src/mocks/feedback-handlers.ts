import { networkDelay } from "./network-delay";
import { http, HttpResponse } from "msw";

export const FEEDBACK_HANDLERS = [
  http.post("*/api/submit-feedback", async () => {
    await networkDelay(1200);
    return HttpResponse.json({
      statusCode: 200,
      body: { message: "Success", link: "fake-url.com", password: "abc123" },
    });
  }),

  http.post("*/api/submit-feedback", async () =>
    HttpResponse.json({ statusCode: 200 }, { status: 200 }),
  ),
];
