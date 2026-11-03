import { describe, expect, it } from "vitest";
import { formatModelPillLabel } from "./format-model-name";

describe("formatModelPillLabel", () => {
  it("keeps the pretty OpenHands label without a free suffix unless backend marks it free", () => {
    expect(formatModelPillLabel("openhands/deepseek-v4.1-flash")).toBe(
      "OpenHands DeepSeek V4.1 Flash",
    );

    expect(
      formatModelPillLabel(
        "openhands/deepseek-v4.1-flash",
        new Set(["openhands/deepseek-v4.1-flash"]),
      ),
    ).toBe("OpenHands DeepSeek V4.1 Flash (free)");
  });
});
