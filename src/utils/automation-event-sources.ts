import type { DeploymentCapabilities } from "#/manifests/types";

/**
 * What Canvas knows about a built-in event source beyond what the automation
 * service reports: the shape of its webhook payload and a trigger that is
 * known to work. The setup form builds its test payload from the shape, and
 * the setup assistant's form tool describes both so a chat request lands on
 * the source's own event keys and a filter that has run, not a guess.
 */
export interface EventSourceGuide {
  label: string;
  /** Top-level payload field that names the event; its value is the event key. */
  eventKeyField: string;
  /** A payload shaped like the source's webhook for `eventKey`. */
  samplePayload: (eventKey: string) => Record<string, unknown>;
  /** A trigger the source's automation template ships, so it has been run. */
  reference?: {
    purpose: string;
    eventKey: string;
    filter: string;
    /** Example literals in `filter` and what a user puts in their place. */
    placeholders: Record<string, string>;
  };
}

// The references are the Data Center templates' triggers
// (OpenHands/extensions `automations/catalog/*-data-center-*`), with their
// form values replaced by the examples the templates' forms show.
export const EVENT_SOURCE_GUIDES: Readonly<Record<string, EventSourceGuide>> = {
  jira_dc: {
    label: "Jira Data Center",
    eventKeyField: "webhookEvent",
    samplePayload: (eventKey) => ({
      webhookEvent: eventKey,
      issue: { key: "PROJ-1", fields: { summary: "Example issue" } },
      comment: { body: "Example comment" },
    }),
    reference: {
      purpose:
        "start when an issue in one project is created with a label or gets that label added",
      eventKey: "jira:issue_*",
      filter:
        "issue.fields.project.key == 'ENG' && ((webhookEvent == 'jira:issue_created' && contains(issue.fields.labels || `[]`, 'openhands')) || (webhookEvent == 'jira:issue_updated' && length(changelog.items[?field == 'labels' && contains(join('', [' ', to_string(toString), ' ']), ' openhands ') && !contains(join('', [' ', to_string(fromString), ' ']), ' openhands ')] || `[]`) > `0`))",
      placeholders: {
        ENG: "the Jira project key",
        openhands: "the label that starts a run",
      },
    },
  },
  bitbucket_data_center: {
    label: "Bitbucket Data Center",
    eventKeyField: "eventKey",
    samplePayload: (eventKey) => ({
      eventKey,
      pullRequest: {
        id: 1,
        title: "Example pull request",
        toRef: {
          displayId: "main",
          repository: { slug: "repo", project: { key: "PROJ" } },
        },
      },
    }),
    reference: {
      purpose:
        "review each non-draft pull request opened or updated in one repository",
      eventKey: "pr:*",
      filter:
        "(eventKey == 'pr:opened' || eventKey == 'pr:from_ref_updated') && pullRequest.draft != `true` && join('/', [pullRequest.toRef.repository.project.key, pullRequest.toRef.repository.slug]) == 'ENG/payments'",
      placeholders: {
        "ENG/payments":
          "the project key and repository slug from the repository's URL",
      },
    },
  },
};

/** The event key patterns a deployment reports for one source, if any. */
function reportedEventKeys(
  capabilities: DeploymentCapabilities,
  source: string,
): string[] | null {
  if (capabilities.eventTypesBySource) {
    return capabilities.eventTypesBySource[source] ?? null;
  }
  // Older services list GitHub's only.
  return source === "github" ? capabilities.eventTypes : null;
}

function describeEventSource(
  capabilities: DeploymentCapabilities,
  source: string,
): string {
  const guide = EVENT_SOURCE_GUIDES[source];
  const keys = reportedEventKeys(capabilities, source);
  const parts = [
    `* ${source}${guide ? ` (${guide.label})` : ""}:`,
    keys?.length
      ? `event keys ${keys.join(", ")}.`
      : "any event key the source delivers.",
  ];
  if (guide) {
    parts.push(`The payload names the event in "${guide.eventKeyField}".`);
  }
  if (guide?.reference) {
    const { purpose, eventKey, filter, placeholders } = guide.reference;
    const replacements = Object.entries(placeholders)
      .map(([literal, meaning]) => `${literal} with ${meaning}`)
      .join(" and ");
    parts.push(
      `Tested trigger to ${purpose}: eventKey "${eventKey}" with eventFilter ${filter}`,
      `Replace ${replacements}; keep the rest of the filter as it is.`,
    );
  }
  return parts.join(" ");
}

/**
 * The event trigger section of the setup assistant's form tool description.
 *
 * The deployment decides which sources exist, so the section is built from
 * its capabilities at conversation start. Without them the assistant is told
 * to ask rather than guess a source.
 */
export function describeEventSources(
  capabilities: DeploymentCapabilities | null | undefined,
): string {
  if (!capabilities) {
    return "This deployment's event sources could not be read. Ask the user which integration delivers the event before filling eventSource, and keep eventKey and eventFilter empty unless they supply them.";
  }
  if (capabilities.eventSources.length === 0) {
    return 'This deployment delivers no webhook events, so event triggers are unavailable: use triggerKind "cron".';
  }
  return [
    "Event sources on this deployment (use exactly these eventSource values):",
    ...capabilities.eventSources.map((source) =>
      describeEventSource(capabilities, source),
    ),
  ].join("\n");
}
