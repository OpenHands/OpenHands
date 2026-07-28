import { describe, expect, it } from "vitest";
import { buildManifestPayload } from "#/manifests/manifest-actions";
import {
  buildRequestBody,
  interpolateText,
} from "#/manifests/manifest-template";
import { createManifest } from "./manifest-test-data";

describe("mapping form values into a request body", () => {
  it("reproduces the request body the service contract expects", () => {
    // The mapping is the highest-consequence part of a manifest: the create
    // endpoint forbids unrecognized keys, so a mismatch is a hard rejection
    // rather than a dropped field. This mirrors the reference manifest and the
    // request body its fixtures pin (OpenHands/extensions#423).

    // Arrange
    const manifest = createManifest({
      submit: {
        action: "automation.create",
        endpoint: { method: "POST", path: "/v1/preset/prompt" },
        payload: {
          name: "PR Reviewer - {{form.repository}}",
          prompt:
            "Review pull requests labeled '{{form.triggerLabel}}' in {{form.repository}}. Review tone: {{form.reviewTone}}.",
          repos: [
            {
              url: "{{form.repository}}",
              ref: "{{form.baseRef}}",
              provider: "github",
            },
          ],
          trigger: {
            type: "cron",
            schedule: "{{form.schedule}}",
            timezone: "{{form.timezone}}",
          },
        },
        onSuccess: { behavior: "navigate", to: "/automations/{{response.id}}" },
        onError: { behavior: "stayOnForm", errorTarget: "field" },
      },
    });
    const values = {
      repository: "OpenHands/agent-server-gui",
      baseRef: "main",
      triggerLabel: "openhands-review",
      reviewTone: "thorough",
      schedule: "*/15 * * * *",
      timezone: "UTC",
    };

    // Act
    const payload = buildManifestPayload(manifest, values);

    // Assert
    expect(payload).toEqual({
      name: "PR Reviewer - OpenHands/agent-server-gui",
      prompt:
        "Review pull requests labeled 'openhands-review' in OpenHands/agent-server-gui. Review tone: thorough.",
      repos: [
        {
          url: "OpenHands/agent-server-gui",
          ref: "main",
          provider: "github",
        },
      ],
      trigger: {
        type: "cron",
        schedule: "*/15 * * * *",
        timezone: "UTC",
      },
    });
  });

  it("has no body to send when setup is handed to a conversation", () => {
    // Arrange
    const manifest = createManifest({
      setupMode: "assisted",
      submit: {
        action: "conversation.start",
        message: "Set up a widget called {{form.widgetName}}.",
        onSuccess: {
          behavior: "navigate",
          to: "/conversations/{{response.conversation_id}}",
        },
        onError: { behavior: "stayOnForm", errorTarget: "form" },
      },
    });

    // Act
    const payload = buildManifestPayload(manifest, { widgetName: "Sprocket" });

    // Assert
    expect(payload).toBeNull();
  });

  it("sends the mapped payload itself when a value is a whole placeholder", () => {
    // Preflight validates what will actually be sent, so `{{submit.payload}}`
    // has to carry the object rather than a stringified copy of it.

    // Arrange
    const payload = { name: "Sprocket", trigger: { type: "cron" } };

    // Act
    const body = buildRequestBody(
      { manifestId: "{{manifest.id}}", draft: "{{submit.payload}}" },
      { manifest: createManifest(), submit: { payload } },
    );

    // Assert
    expect(body).toEqual({ manifestId: "widget-setup", draft: payload });
  });

  it("renders an unfilled value as blank so the caller can supply its own fallback", () => {
    // Act
    const text = interpolateText("Runs on {{form.schedule}}", { form: {} });

    // Assert
    expect(text).toBe("Runs on ");
  });
});
