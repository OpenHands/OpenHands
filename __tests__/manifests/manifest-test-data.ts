import type { SetupBlock, SetupEntry } from "#/manifests/types";

/**
 * A minimal catalog entry the host will admit.
 *
 * Tests override only the part they exercise. The three published entries live
 * in `automation-setup.test.ts`, where the derived request bodies are pinned
 * against the contract fixtures.
 */
export function createSetupEntry(
  overrides: Partial<SetupEntry> = {},
): SetupEntry {
  return {
    id: "widget-monitor",
    name: "Widget monitor",
    description: "Watch widgets and report on them.",
    requires: {
      integrations: { github: { message: "Used to read widgets." } },
    },
    setup: createSetup(),
    ...overrides,
  };
}

export function createSetup(overrides: Partial<SetupBlock> = {}): SetupBlock {
  return {
    version: "1.0",
    mode: "direct",
    form: {
      triggers: {
        cron: {
          schedule: {
            type: "cron",
            label: "Check frequency",
            help: "How often to look.",
            default: "*/15 * * * *",
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
        widgetName: {
          type: "text",
          label: "Widget name",
          help: "What to call it.",
          required: true,
        },
      },
    },
    prompt: "Report on {{form.widgetName}} in {{form.repository}}.",
    ...overrides,
  };
}

/** Build a candidate that is invalid in one specific way. */
export function createSetupEntryWith(
  overrides: Record<string, unknown>,
): unknown {
  return { ...createSetupEntry(), ...overrides };
}
