import { test, expect } from "vitest";
import { getProviderId } from "../../src/utils/map-provider";

test("getProviderId round-trips the display name back to the provider id", () => {
  // The backend addresses providers by their litellm id (`perplexity`), so the
  // display name must map back to that id, not to a misspelling of it.
  expect(getProviderId("Perplexity AI")).toBe("perplexity");
  expect(getProviderId("OpenAI")).toBe("openai");
  expect(getProviderId("Some New Provider")).toBe("Some New Provider");
});
