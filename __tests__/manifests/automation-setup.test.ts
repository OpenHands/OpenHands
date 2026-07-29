import { describe, expect, it, vi } from "vitest";
import {
  buildAssistedMessage,
  buildCreatePayload,
  buildPreflightBody,
  deriveErrorMap,
} from "#/manifests/automation-setup";
import { validateSetupEntry } from "#/manifests/manifest-validation";
import type { SetupEntry } from "#/manifests/types";

// The command a skill publishes in its own frontmatter, which the host looks up
// rather than storing. Pinned here so the assertion does not move when the
// packaged catalog does.
vi.mock("@openhands/extensions/skills", () => ({
  SKILLS_CATALOG: [
    {
      name: "incident-retrospective",
      description: "Draft an incident retrospective.",
      triggers: ["/incident-retro:setup"],
      content: "",
    },
  ],
}));

/**
 * The setup blocks published in `OpenHands/extensions`, and the request bodies
 * its contract fixtures say they must produce. Those bodies were verified
 * against the live service, and the create model forbids extra keys, so any
 * divergence here is a 422 in production rather than a cosmetic difference.
 */
const PR_REVIEWER: SetupEntry = {
  id: "github-pr-reviewer",
  name: "GitHub Code Review Agent",
  description: "Watch for a configurable label on GitHub pull requests.",
  requires: {
    integrations: {
      github: { message: "Used to read pull requests and post comments." },
    },
    features: ["repoClone", "presetPrompt"],
  },
  setup: {
    version: "1.0",
    mode: "direct",
    form: {
      triggers: {
        cron: {
          schedule: {
            type: "cron",
            label: "Check frequency",
            help: "How often to look for newly labelled pull requests.",
            default: "*/15 * * * *",
            required: true,
          },
          timezone: {
            type: "timezone",
            label: "Timezone",
            help: "Timezone the schedule is interpreted in.",
            default: "UTC",
            required: true,
          },
        },
      },
      args: {
        repository: {
          type: "repo-picker",
          label: "Repository",
          help: "The repository whose pull requests will be reviewed.",
          provider: "github",
          required: true,
        },
        triggerLabel: {
          type: "text",
          label: "Trigger label",
          help: "Only pull requests carrying this label are reviewed.",
          default: "openhands-review",
          required: true,
          constraints: { minLength: 1, maxLength: 50 },
        },
        reviewTone: {
          type: "select",
          label: "Review tone",
          help: "How detailed the review comments should be.",
          default: "concise",
          required: true,
          options: [
            { value: "concise", label: "Concise" },
            { value: "thorough", label: "Thorough" },
          ],
        },
      },
    },
    prompt:
      "Review pull requests labeled '{{form.triggerLabel}}' in {{form.repository}}. Review tone: {{form.reviewTone}}.",
  },
};

const REPO_MONITOR: SetupEntry = {
  id: "github-repo-monitor",
  name: "GitHub repository monitor",
  description: "Watch a repository for @OpenHands mentions.",
  requires: {
    integrations: {
      github: { message: "Used to read comments and post replies." },
    },
    features: ["repoClone", "presetPrompt", "webhookDelivery"],
  },
  setup: {
    version: "1.0",
    mode: "direct",
    form: {
      triggers: {
        event: {
          on: {
            type: "select",
            label: "Respond to",
            help: "Which GitHub comment event starts a conversation.",
            default: "issue_comment.created",
            required: true,
            options: [
              {
                value: "issue_comment.created",
                label: "Comments on issues and pull requests",
              },
              {
                value: "pull_request_review_comment.created",
                label: "Pull request review comments only",
              },
            ],
          },
          triggerPhrase: {
            type: "text",
            label: "Trigger phrase",
            help: "Only comments containing this phrase start a conversation.",
            default: "@openhands",
            required: true,
            constraints: {
              minLength: 2,
              maxLength: 50,
              format: "safeExpressionLiteral",
            },
          },
        },
      },
      args: {
        repository: {
          type: "repo-picker",
          label: "Repository",
          help: "The repository or repositories to watch for mentions.",
          provider: "github",
          required: true,
        },
        ref: {
          type: "text",
          label: "Base branch",
          help: "Branch checked out when the agent responds.",
          default: "main",
          required: true,
          constraints: { minLength: 1, maxLength: 255 },
        },
      },
    },
    prompt:
      "A comment in {{form.repository}} mentions '{{form.triggerPhrase}}'. Read the surrounding issue or pull request context from the event payload, then post a helpful reply as a comment on the same thread.",
    filter:
      "icontains(comment.body, '{{form.triggerPhrase}}') && glob(repository.full_name, '{{form.repository}}')",
  },
};

const RETRO_DRAFTER: SetupEntry = {
  id: "incident-retrospective-drafter",
  name: "Incident retrospective drafter",
  description: "Collect incident chatter and draft a timeline.",
  requires: {
    integrations: {
      slack: { message: "Reads incident discussion and timestamps." },
      linear: { message: "Reads follow-up tickets, owners, and status." },
      notion: {
        message: "Publishes the drafted retrospective.",
        required: false,
      },
    },
    features: ["mcpTools", "conversationDispatch"],
  },
  skill: "incident-retrospective",
  setup: {
    version: "1.0",
    mode: "assisted",
    form: {
      note: "These answers start the conversation.",
      args: {
        incidentChannel: {
          type: "text",
          label: "Incident channel",
          help: "The Slack channel where incidents are discussed.",
          required: false,
          constraints: { maxLength: 100 },
        },
        linearTeam: {
          type: "text",
          label: "Linear team",
          help: "The team whose follow-up tickets belong in the retrospective.",
          required: false,
          constraints: { maxLength: 100 },
        },
        notionDestination: {
          type: "text",
          label: "Notion destination",
          help: "Page or database where drafts should be published.",
          required: false,
          constraints: { maxLength: 200 },
        },
        triggerPreference: {
          type: "select",
          label: "How should it run?",
          help: "A starting preference only.",
          default: "undecided",
          required: false,
          options: [
            { value: "undecided", label: "Let the agent recommend one" },
            { value: "scheduled", label: "On a schedule" },
          ],
        },
        notes: {
          type: "textarea",
          label: "Anything else the agent should know?",
          help: "Incident naming conventions, who should be notified.",
          required: false,
          constraints: { maxLength: 2000 },
        },
      },
    },
    message:
      "The user supplied these starting points. Confirm each one, ask about anything left blank, then create the automation.\n\n- Incident channel: {{form.incidentChannel}}\n- Linear team: {{form.linearTeam}}\n- Notion destination: {{form.notionDestination}}\n- Preferred trigger: {{form.triggerPreference}}\n- Notes: {{form.notes}}\n\nStill undecided and required before creation: the incident identifier format and the approved Notion retrospective template.",
  },
};

const PR_REVIEWER_VALUES = {
  repository: "OpenHands/agent-server-gui",
  triggerLabel: "openhands-review",
  reviewTone: "thorough",
  schedule: "*/15 * * * *",
  timezone: "UTC",
};

const PR_REVIEWER_DRAFT = {
  name: "GitHub Code Review Agent - OpenHands/agent-server-gui",
  prompt:
    "Review pull requests labeled 'openhands-review' in OpenHands/agent-server-gui. Review tone: thorough.",
  repos: [{ url: "OpenHands/agent-server-gui", provider: "github" }],
  trigger: { type: "cron", schedule: "*/15 * * * *", timezone: "UTC" },
};

describe("buildCreatePayload", () => {
  it("derives the scheduled request body the contract fixture pins", () => {
    // Act
    const payload = buildCreatePayload(PR_REVIEWER, PR_REVIEWER_VALUES);

    // Assert
    expect(payload).toEqual(PR_REVIEWER_DRAFT);
  });

  it("derives an event trigger's source and filter, which no manifest states", () => {
    // Act
    const payload = buildCreatePayload(REPO_MONITOR, {
      repository: "OpenHands/agent-server-gui",
      triggerPhrase: "@openhands",
      on: "issue_comment.created",
      ref: "main",
    });

    // Assert
    expect(payload).toEqual({
      name: "GitHub repository monitor - OpenHands/agent-server-gui",
      prompt:
        "A comment in OpenHands/agent-server-gui mentions '@openhands'. Read the surrounding issue or pull request context from the event payload, then post a helpful reply as a comment on the same thread.",
      repos: [
        {
          url: "OpenHands/agent-server-gui",
          ref: "main",
          provider: "github",
        },
      ],
      trigger: {
        type: "event",
        source: "github",
        on: "issue_comment.created",
        filter:
          "icontains(comment.body, '@openhands') && glob(repository.full_name, 'OpenHands/agent-server-gui')",
      },
    });
  });

  it("sends no request body for an entry that hands setup to a conversation", () => {
    // Act
    const payload = buildCreatePayload(RETRO_DRAFTER, {});

    // Assert
    expect(payload).toBeNull();
  });
});

describe("buildPreflightBody", () => {
  it("wraps the draft in the envelope the validate endpoint expects", () => {
    // Act
    const body = buildPreflightBody(PR_REVIEWER, PR_REVIEWER_VALUES);

    // Assert
    expect(body).toEqual({
      automationId: "github-pr-reviewer",
      endpoint: "/v1/preset/prompt",
      draft: PR_REVIEWER_DRAFT,
    });
  });
});

describe("deriveErrorMap", () => {
  it("recovers which fields built each payload path", () => {
    // Act
    const errorMap = deriveErrorMap(PR_REVIEWER);

    // Assert
    expect(errorMap).toEqual({
      name: ["repository"],
      prompt: ["triggerLabel", "repository", "reviewTone"],
      "repos[0].url": ["repository"],
      "trigger.schedule": ["schedule"],
      "trigger.timezone": ["timezone"],
    });
  });
});

describe("buildAssistedMessage", () => {
  it("opens with the owning skill's command and the interpolated answers", () => {
    // Act
    const message = buildAssistedMessage(RETRO_DRAFTER, {
      incidentChannel: "#incidents",
      linearTeam: "Reliability",
      notionDestination: "",
      triggerPreference: "undecided",
      notes: "",
    });

    // Assert
    expect(message).toBe(
      "/incident-retro:setup\n\nThe user supplied these starting points. Confirm each one, ask about anything left blank, then create the automation.\n\n- Incident channel: #incidents\n- Linear team: Reliability\n- Notion destination: \n- Preferred trigger: undecided\n- Notes: \n\nStill undecided and required before creation: the incident identifier format and the approved Notion retrospective template.",
    );
  });
});

describe("the published manifests", () => {
  // The host enforces invariants its own derivation depends on, which the
  // published schema does not. Every shipped entry has to satisfy them too.
  it.each([
    ["github-pr-reviewer", PR_REVIEWER],
    ["github-repo-monitor", REPO_MONITOR],
    ["incident-retrospective-drafter", RETRO_DRAFTER],
  ])("admits %s", (_id, entry) => {
    // Act
    const result = validateSetupEntry(entry);

    // Assert
    expect(result).toEqual({ valid: true, errors: [] });
  });
});
