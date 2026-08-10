import { describe, expect, it } from "vitest";
import { validateSetupEntry } from "#/manifests/manifest-validation";
import {
  createSetup,
  createSetupEntry,
  createSetupEntryWith,
} from "./manifest-test-data";

describe("validateSetupEntry", () => {
  it("admits a well-formed manifest", () => {
    // Arrange
    const entry = createSetupEntry();

    // Act
    const result = validateSetupEntry(entry);

    // Assert
    expect(result).toEqual({ valid: true, errors: [] });
  });

  it("admits a direct entry carrying a fallback-conversation message", () => {
    // Arrange — the seed for the conversation offered when the deployment
    // cannot run the direct path.
    const entry = createSetupEntry({
      setup: createSetup({
        message: "Set this up in a conversation instead.",
      }),
    });

    // Act
    const result = validateSetupEntry(entry);

    // Assert
    expect(result).toEqual({ valid: true, errors: [] });
  });

  // Each case is a separate invariant the host enforces on data authored in
  // another repository. A manifest that trips any of them must not render.
  it.each([
    [
      "a setup version this host cannot interpret",
      { setup: createSetup({ version: "2.0" as "1.0" }) },
    ],
    [
      "markup inside user-visible copy",
      { description: "<img src=x onerror=alert(1)>" },
    ],
    [
      "a placeholder namespace the host does not expose",
      { setup: createSetup({ prompt: "Use {{secrets.githubToken}}." }) },
    ],
    [
      // A direct entry may carry a fallback-conversation seed, but it is still
      // copy, held to the same injection rules as an assisted message.
      "markup inside a direct entry's fallback message",
      { setup: createSetup({ message: "<img src=x onerror=alert(1)>" }) },
    ],
    [
      // The host reads one trigger kind to build the request, so a second one
      // would be silently dropped rather than refused.
      "more trigger kinds than the host can send",
      {
        setup: createSetup({
          form: {
            triggers: {
              cron: {
                schedule: {
                  type: "cron",
                  label: "Frequency",
                  help: "How often.",
                  required: true,
                },
              },
              event: {
                on: {
                  type: "select",
                  label: "Respond to",
                  help: "Which event.",
                  required: true,
                  options: [{ value: "push", label: "Push" }],
                },
              },
            },
            args: {
              repository: {
                type: "repo-picker",
                label: "Repository",
                help: "Which repository.",
                provider: "github",
                required: true,
              },
            },
          },
        }),
      },
    ],
    [
      // An event trigger's source is read off the repository field's provider.
      "an event trigger with no repository field to take its source from",
      {
        setup: createSetup({
          form: {
            triggers: {
              event: {
                on: {
                  type: "select",
                  label: "Respond to",
                  help: "Which event.",
                  required: true,
                  options: [{ value: "push", label: "Push" }],
                },
              },
            },
            args: {
              widgetName: {
                type: "text",
                label: "Widget name",
                help: "What to call it.",
                required: true,
              },
            },
          },
          filter: "icontains(body, '{{form.widgetName}}')",
        }),
      },
    ],
    [
      // Both halves merge into one value map, so a repeat would shadow a field
      // and misaddress every error reported against it.
      "a field name declared in both halves of the form",
      {
        setup: createSetup({
          form: {
            triggers: {
              cron: {
                repository: {
                  type: "text",
                  label: "Repository",
                  help: "Shadows the argument below.",
                  required: true,
                },
              },
            },
            args: {
              repository: {
                type: "repo-picker",
                label: "Repository",
                help: "Which repository to watch.",
                provider: "github",
                required: true,
              },
            },
          },
        }),
      },
    ],
  ])("refuses %s", (_case, overrides) => {
    // Arrange
    const candidate = createSetupEntryWith(overrides);

    // Act
    const result = validateSetupEntry(candidate);

    // Assert
    expect(result.valid).toBe(false);
  });

  it("reports every problem at once so an author sees the whole picture", () => {
    // Arrange
    const candidate = createSetupEntryWith({ name: "", description: "" });

    // Act
    const { errors } = validateSetupEntry(candidate);

    // Assert
    expect(errors).toHaveLength(2);
  });
});
