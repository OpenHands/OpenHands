import { beforeEach, describe, expect, it } from "vitest";

import {
  clearAutomationFormSession,
  getAutomationFormSession,
  initializeAutomationFormSession,
  patchAutomationFormSession,
} from "#/api/automation-form-session";

describe("automation-form-session", () => {
  beforeEach(() => {
    clearAutomationFormSession("test-conv-1");
  });

  it("preserves existingCustomTarballPath across initialization and normalized updates", () => {
    initializeAutomationFormSession("test-conv-1", {
      prompt: "",
      kind: "custom",
      editingAutomationId: "auto-1",
      existingCustomTarballPath: "oh-internal://uploads/saved-bundle.tar.gz",
      form: {
        name: "My Automation",
        entrypoint: "python3 main.py",
      },
    });

    const initialSession = getAutomationFormSession("test-conv-1");
    expect(initialSession?.existingCustomTarballPath).toBe(
      "oh-internal://uploads/saved-bundle.tar.gz",
    );

    // Perform an unrelated patch update
    patchAutomationFormSession(
      "test-conv-1",
      { name: "Updated Automation Title" },
      { source: "user" },
    );

    const patchedSession = getAutomationFormSession("test-conv-1");
    expect(patchedSession?.existingCustomTarballPath).toBe(
      "oh-internal://uploads/saved-bundle.tar.gz",
    );
    expect(patchedSession?.form?.name).toBe("Updated Automation Title");
  });
});
