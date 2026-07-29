import { describe, it, vi, beforeEach, afterEach } from "vitest";

// Mock the store module
vi.mock("#/store", () => ({
  default: {
    dispatch: vi.fn(),
  },
}));

describe("handleObservationMessage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.resetAllMocks();
  });

  it.todo("updates browser state when receiving a browse observation");

  it.todo(
    "updates browser state when receiving a browse_interactive observation",
  );
});
