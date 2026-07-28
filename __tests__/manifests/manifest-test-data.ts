import type { ExtensionManifest } from "#/manifests/types";

/**
 * A minimal manifest the host will admit. Deliberately not an automation: the
 * host is only allowed to be a generic renderer, so its tests are written
 * against a manifest that carries no automation vocabulary.
 *
 * Tests override only the part they exercise.
 */
export function createManifest(
  overrides: Partial<ExtensionManifest> = {},
): ExtensionManifest {
  return {
    manifestVersion: "1.0",
    id: "widget-setup",
    name: "Widget setup",
    category: "Demo",
    description: "Configure a widget.",
    setupMode: "direct",
    routes: [{ path: "/ext/widget", page: "setup" }],
    form: {
      fields: [
        {
          name: "widgetName",
          type: "text",
          label: "Widget name",
          help: "What to call it.",
          required: true,
        },
      ],
    },
    validation: {
      onInvalid: { behavior: "blockSubmit", errorTarget: "field" },
    },
    review: {
      title: "Review",
      summary: [{ label: "Name", value: "{{form.widgetName}}" }],
      confirmLabel: "Create",
    },
    submit: {
      action: "automation.create",
      endpoint: { method: "POST", path: "/v1/preset/prompt" },
      payload: { name: "{{form.widgetName}}" },
      onSuccess: { behavior: "navigate", to: "/widgets/{{response.id}}" },
      onError: { behavior: "stayOnForm", errorTarget: "field" },
    },
    analytics: {
      consent: "required",
      stages: [
        {
          id: "widget_setup_opened",
          on: "route.entered",
          properties: { manifest_id: "{{manifest.id}}" },
        },
      ],
    },
    ...overrides,
  };
}

/** Build a candidate that is invalid in one specific way. */
export function createManifestWith(
  overrides: Record<string, unknown>,
): unknown {
  return { ...createManifest(), ...overrides };
}
