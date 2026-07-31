import type {
  InterfaceManifest,
  SetupBlock,
  SetupEntry,
} from "#/manifests/types";

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

/**
 * A minimal interface manifest the host will admit. Featured ids are real
 * catalog entries, because admission checks them against the published
 * catalog. Tests override only the part they exercise.
 */
export function createInterfaceManifest(
  overrides: Partial<InterfaceManifest> = {},
): InterfaceManifest {
  return {
    version: "1.0",
    routes: {
      list: "/automations",
      setup: "/automations/new/:automationId",
      detail: "/automations/:automationId",
    },
    navigation: {
      sidebar: { label: "Automate everything" },
      commandMenu: {
        title: "Automation center",
        description: "Review the widget automations.",
        keywords: "widgets cron",
      },
    },
    pages: {
      list: { title: "Widget automations", subtitle: "All of the widgets." },
      detail: { backLabel: "Back to the widgets" },
    },
    docsUrl: "https://docs.openhands.dev/widgets",
    edit: {
      title: "Edit the widget automation",
      fields: {
        name: { type: "text", label: "Widget name", required: true },
        timeout: {
          type: "number",
          label: "Widget timeout",
          required: false,
          constraints: { min: 1, max: 900 },
        },
      },
    },
    importExport: {
      fileKind: "automation",
      fileVersion: 1,
      filenameSuffix: ".automation.json",
      importDefaults: {
        repoProvider: "github",
        placeholderEventSource: "agent-canvas-import",
      },
    },
    endpoints: {
      list: "/v1",
      detail: "/v1/{id}",
      dispatch: "/v1/{id}/dispatch",
      runs: "/v1/{id}/runs",
      tarball: "/v1/{id}/tarball",
      health: "/health",
      capabilities: "/v1/capabilities",
      validate: "/v1/validate",
      createPrompt: "/v1/preset/prompt",
      createPlugin: "/v1/preset/plugin",
    },
    featuredAutomationIds: ["github-pr-reviewer"],
    responderIntegrationIds: ["github", "slack"],
    ...overrides,
  };
}

/** Build an interface candidate that is invalid in one specific way. */
export function createInterfaceManifestWith(
  overrides: Record<string, unknown>,
): unknown {
  return { ...createInterfaceManifest(), ...overrides };
}
