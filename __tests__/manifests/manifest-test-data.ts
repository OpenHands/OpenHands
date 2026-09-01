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
      edit: { title: "Edit the widget automation" },
    },
    docsUrl: "https://docs.openhands.dev/widgets",
    attributes: {
      name: { type: "text", label: "Widget name", required: true },
      timeout: {
        type: "number",
        label: "Widget timeout",
        required: false,
        constraints: { min: 1, max: 900 },
      },
    },
    // Every importExport datum differs from the host default, so a test that
    // reads it back can tell manifest data from fallback data.
    importExport: {
      fileKind: "widget",
      fileVersion: 1,
      filenameSuffix: ".widget.json",
      importDefaults: {
        repoProvider: "gitlab",
        placeholderEventSource: "widget-import",
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
    responderIntegrationIds: ["github"],
    ...overrides,
  };
}

/** Build an interface candidate that is invalid in one specific way. */
export function createInterfaceManifestWith(
  overrides: Record<string, unknown>,
): unknown {
  return { ...createInterfaceManifest(), ...overrides };
}

/**
 * The factory manifest plus a complete sub-page surface. Like the base
 * factory, every caption differs from the production manifest and from any
 * host string, so a test that reads one back knows where it came from.
 */
export function createInterfaceManifestWithSubPages(): InterfaceManifest {
  const base = createInterfaceManifest();
  return {
    ...base,
    routes: { ...base.routes, templates: "/automations/templates" },
    navigation: {
      ...base.navigation,
      subPages: [
        { page: "list", label: "Widget dashboard", icon: "layout-dashboard" },
        { page: "templates", label: "Widget templates", icon: "sparkles" },
      ],
    },
    pages: {
      ...base.pages,
      list: {
        ...base.pages.list,
        overview: {
          label: "Widget overview",
          tiles: [
            {
              metric: "automations",
              label: "Widget count",
              detail: "{{active}} live",
              icon: "bot",
            },
            {
              metric: "needs-attention",
              label: "Widget attention",
              detail: "Broken widgets",
              zeroDetail: "No broken widgets",
              icon: "circle-alert",
            },
            {
              metric: "total-runs",
              label: "Widget runs",
              detail: "Counted so far",
              icon: "activity",
            },
            {
              metric: "average-duration",
              label: "Widget pace",
              detail: "Recent finished runs",
              icon: "timer",
            },
          ],
        },
        filters: [
          {
            id: "status",
            label: "Filter widgets by state",
            options: [
              { value: "all", label: "Every state" },
              { value: "active", label: "Live" },
              { value: "failing", label: "Broken" },
              { value: "disabled", label: "Off" },
            ],
          },
          {
            id: "trigger",
            label: "Filter widgets by trigger",
            options: [
              { value: "all", label: "Every trigger" },
              { value: "schedule", label: "Timed" },
              { value: "event", label: "Evented" },
            ],
          },
        ],
        sort: {
          label: "Order widgets",
          default: "name",
          options: [
            { value: "last-run", label: "Latest" },
            { value: "runs", label: "Busiest" },
            { value: "name", label: "Alphabetical" },
          ],
        },
        insights: {
          health: {
            healthy: "Fine",
            failing: "Broken",
            running: "Busy",
            disabled: "Off",
            neverRun: "Fresh",
            checking: "Looking",
          },
          lastRun: {
            label: "Last widget run",
            never: "Not yet",
            justNow: "Moments ago",
          },
          stats: {
            runs: "Widget runs",
            recentSuccess: "Widget wins",
            averageDuration: "Widget pace",
          },
        },
      },
      templates: {
        title: "Widget templates",
        description: "Pick a proven widget to start from.",
      },
    },
  };
}
