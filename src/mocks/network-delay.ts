import { delay } from "msw";

// Demo latency makes loading states visible; unit tests need only an async turn.
export const networkDelay = (duration?: number) =>
  delay(import.meta.env.MODE === "test" ? 0 : duration);
