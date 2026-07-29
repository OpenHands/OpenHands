import { describe, it, expect, beforeEach } from "vitest";
import { useInitialQueryStore } from "../src/stores/initial-query-store";

describe("Initial Query Behavior", () => {
  beforeEach(() => {
    // Reset the store before each test
    useInitialQueryStore.getState().reset();
  });

  it("should clear initial query when clearInitialPrompt is called", () => {
    const { setInitialPrompt, clearInitialPrompt, initialPrompt } =
      useInitialQueryStore.getState();

    // Set up initial query in the store
    setInitialPrompt("test query");
    expect(useInitialQueryStore.getState().initialPrompt).toBe("test query");

    // Clear the initial query
    clearInitialPrompt();

    // Verify initial query is cleared
    expect(useInitialQueryStore.getState().initialPrompt).toBeNull();
  });
});
