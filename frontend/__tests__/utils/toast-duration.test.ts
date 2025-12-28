import { describe, it, expect } from "vitest";
import { calculateToastDuration } from "#/utils/toast-duration";

describe("calculateToastDuration", () => {
  it("should return minimum duration for short messages", () => {
    const shortMessage = "OK";
    const duration = calculateToastDuration(shortMessage, 5000);
    expect(duration).toBe(5000);
  });

  it("should return minimum duration for messages that calculate below minimum", () => {
    const shortMessage = "Settings saved";
    const duration = calculateToastDuration(shortMessage, 5000);
    expect(duration).toBe(5000);
  });

  it("should calculate longer duration for long messages", () => {
    const longMessage =
      "Settings saved. For old conversations, you will need to stop and restart the conversation to see the changes.";
    const duration = calculateToastDuration(longMessage, 5000);
    expect(duration).toBeGreaterThan(5000);
    expect(duration).toBeLessThanOrEqual(10000);
  });

  it("should respect maximum duration cap", () => {
    const veryLongMessage = "A".repeat(10000); // Very long message
    const duration = calculateToastDuration(veryLongMessage, 5000, 10000);
    expect(duration).toBe(10000);
  });

  it("should use custom minimum and maximum durations", () => {
    const message = "Test message";
    const customMin = 3000;
    const customMax = 8000;
    const duration = calculateToastDuration(message, customMin, customMax);
    expect(duration).toBeGreaterThanOrEqual(customMin);
    expect(duration).toBeLessThanOrEqual(customMax);
  });

  it("should calculate duration based on reading speed", () => {
    // Test with a message that should take exactly the calculated time
    // At 200 WPM (1000 chars/min), 60 chars should take 3.6 seconds
    // With 1.5x buffer, that's 5.4 seconds
    const message = "This is a test message that contains exactly sixty chars.";
    expect(message.length).toBe(57); // Close to 60 chars

    const duration = calculateToastDuration(message, 0, 20000); // No min/max constraints

    // Should be around 5.4 seconds (5400ms) for 57 characters
    expect(duration).toBeGreaterThan(5000);
    expect(duration).toBeLessThan(6000);
  });
});
