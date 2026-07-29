import { describe, expect, it } from "vitest";
import {
  mapServiceErrors,
  normalizeServiceErrors,
} from "#/manifests/manifest-error-map";

/** The body the host derived, as the service received it. */
const PAYLOAD = {
  name: "PR Reviewer - OpenHands/agent-server-gui",
  prompt: "Review pull requests labeled 'openhands-review'.",
  repos: [{ url: "OpenHands/agent-server-gui", ref: "main" }],
  trigger: { type: "cron", schedule: "0 0 31 2 *", timezone: "UTC" },
};

/** As `deriveErrorMap` produces it: every path maps to the fields that built it. */
const ERROR_MAP = {
  "repos[0].ref": ["ref"],
  "trigger.schedule": ["schedule"],
  prompt: ["triggerLabel", "reviewTone"],
};

describe("normalizeServiceErrors", () => {
  it("reads field-addressed errors out of a preflight result", () => {
    // Act
    const errors = normalizeServiceErrors(
      {
        valid: false,
        errors: [
          {
            field: "trigger.schedule",
            message: "Minimum interval for this deployment is 5 minutes.",
          },
        ],
      },
      PAYLOAD,
    );

    // Assert
    expect(errors).toEqual([
      {
        path: "trigger.schedule",
        message: "Minimum interval for this deployment is 5 minutes.",
      },
    ]);
  });

  it("resolves a rejection addressed through a discriminated union", () => {
    // The service names the branch it validated against — `trigger.cron` —
    // but the body it received has no `cron` key. Walking the payload and
    // skipping segments that do not resolve reduces the address to one the
    // manifest can map, without the host knowing what a trigger is.

    // Act
    const errors = normalizeServiceErrors(
      {
        detail: [
          {
            loc: ["body", "trigger", "cron", "schedule"],
            msg: "Cron expression cannot produce any future fire times.",
          },
        ],
      },
      PAYLOAD,
    );

    // Assert
    expect(errors[0].path).toBe("trigger.schedule");
  });

  it("addresses an element of a list by its index", () => {
    // Act
    const errors = normalizeServiceErrors(
      { detail: [{ loc: ["body", "repos", 0, "ref"], msg: "Unknown ref." }] },
      PAYLOAD,
    );

    // Assert
    expect(errors[0].path).toBe("repos[0].ref");
  });
});

describe("mapServiceErrors", () => {
  it("highlights the field that produced the rejected value", () => {
    // Act
    const { fieldErrors } = mapServiceErrors(
      [{ path: "trigger.schedule", message: "Too frequent." }],
      ERROR_MAP,
    );

    // Assert
    expect(fieldErrors).toEqual({ schedule: "Too frequent." });
  });

  it("highlights every field that contributed to a combined value", () => {
    // Act
    const { fieldErrors } = mapServiceErrors(
      [{ path: "prompt", message: "Prompt is too long." }],
      ERROR_MAP,
    );

    // Assert
    expect(fieldErrors).toEqual({
      triggerLabel: "Prompt is too long.",
      reviewTone: "Prompt is too long.",
    });
  });

  it("surfaces an unmappable rejection against the form rather than losing it", () => {
    // Act
    const { fieldErrors, formErrors } = mapServiceErrors(
      [{ path: "tarball_path", message: "Upload failed." }],
      ERROR_MAP,
    );

    // Assert
    expect({ fieldErrors, formErrors }).toEqual({
      fieldErrors: {},
      formErrors: ["Upload failed."],
    });
  });
});
