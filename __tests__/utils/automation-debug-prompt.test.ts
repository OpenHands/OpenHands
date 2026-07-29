import { describe, it, expect } from "vitest";
import { buildAutomationDebugPrompt } from "#/utils/automation-debug-prompt";

describe("buildAutomationDebugPrompt", () => {
  it("includes the automation name, its instructions, the stderr, and the run id", () => {
    // Arrange
    const input = {
      automationName: "Daily Jira digest",
      automationPrompt: "Summarize new Jira issues each morning",
      errorDetail: "Process exited with code 1",
      stderr: "Traceback ...\nHTTP Error 410: Gone",
      runId: "run-42",
    };

    // Act
    const prompt = buildAutomationDebugPrompt(input);

    // Assert
    expect(prompt).toContain("Daily Jira digest");
    expect(prompt).toContain("Summarize new Jira issues each morning");
    expect(prompt).toContain("HTTP Error 410: Gone");
    expect(prompt).toContain("run-42");
    // The live stderr is preferred over the run-level error_detail.
    expect(prompt).not.toContain("Process exited with code 1");
  });

  it("falls back to error_detail when stderr is blank", () => {
    // Arrange — sandbox gone, so no stderr was fetched.
    const prompt = buildAutomationDebugPrompt({
      automationName: "Daily Jira digest",
      automationPrompt: "Summarize new Jira issues",
      errorDetail: "Process exited with code 1",
      stderr: "   \n  ",
      runId: "run-42",
    });

    // Assert
    expect(prompt).toContain("Process exited with code 1");
    expect(prompt).not.toContain("No error output");
  });

  it("omits the instructions section when the automation prompt is absent", () => {
    // Act
    const prompt = buildAutomationDebugPrompt({
      automationName: "Daily Jira digest",
      automationPrompt: null,
      errorDetail: "boom",
      stderr: "",
      runId: "run-42",
    });

    // Assert
    expect(prompt).not.toContain("What the automation was set up to do");
  });

  it("reports missing output when neither stderr nor error_detail is available", () => {
    // Act
    const prompt = buildAutomationDebugPrompt({
      automationName: "Daily Jira digest",
      automationPrompt: "Summarize new Jira issues",
      errorDetail: null,
      stderr: "",
      runId: "run-42",
    });

    // Assert
    expect(prompt).toContain("No error output was captured");
  });

  it("truncates oversized stderr, keeping the tail", () => {
    // Arrange — an error stream well over the 4000-char cap, with distinct
    // head and tail markers so we can prove which end survives.
    const stderr = `HEAD_MARKER${"x".repeat(5000)}TAIL_MARKER`;

    // Act
    const prompt = buildAutomationDebugPrompt({
      automationName: "Daily Jira digest",
      automationPrompt: "Summarize new Jira issues",
      errorDetail: null,
      stderr,
      runId: "run-42",
    });

    // Assert — the tail (the actual error) is kept, the head is dropped.
    expect(prompt).toContain("TAIL_MARKER");
    expect(prompt).toContain("…(truncated)…");
    expect(prompt).not.toContain("HEAD_MARKER");
  });
});
