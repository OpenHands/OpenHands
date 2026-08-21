import { describe, expect, it, vi } from "vitest";
import incidentFixture from "@openhands/extensions/testing/automations/incident-retrospective-drafter.json";
import prReviewerFixture from "@openhands/extensions/testing/automations/github-pr-reviewer.json";
import repoMonitorFixture from "@openhands/extensions/testing/automations/github-repo-monitor.json";
import {
  automationCreateEndpoint,
  buildAssistedMessage,
  buildCreatePayload,
  buildPreflightBody,
  deriveErrorMap,
} from "#/manifests/automation-setup";
import {
  mapServiceErrors,
  normalizeServiceErrors,
} from "#/manifests/manifest-error-map";
import { validateFormValues } from "#/manifests/manifest-local-validation";
import { SETUP_REGISTRY } from "#/manifests/manifest-sources";
import type { SetupEntry, SetupFormValues } from "#/manifests/types";
import { createSetup, createSetupEntry } from "./manifest-test-data";

// The one word of a derived name the host writes rather than reads off the
// entry is translated, and the derivation runs where no translator can be
// passed in, so it reads the shared instance. Rendered as `en` does, because
// the fixtures pin the sentence the service was sent; the spy is what pins the
// key, so both halves stay covered.
const { translate } = vi.hoisted(() => ({
  translate: vi.fn(
    (_key: string, options: Record<string, unknown>) =>
      `${options.total} repositories`,
  ),
}));
vi.mock("#/i18n", () => ({ default: { t: translate } }));

// The command a skill publishes in its own frontmatter, which the host looks
// up rather than storing. Pinned so the assertion does not move when the
// packaged skills catalog does; the automation catalog itself is imported for
// real, because pinning its derivation to the contract fixtures published
// beside it is this file's point.
vi.mock("@openhands/extensions/skills", () => ({
  SKILLS_CATALOG: [
    {
      name: "incident-retrospective",
      description: "Draft an incident retrospective.",
      triggers: ["/incident-retro:setup"],
      content: "",
    },
    {
      name: "github-repo-monitor",
      description: "Watch a GitHub repository for mentions.",
      triggers: ["/github-monitor:poll"],
      content: "",
    },
  ],
}));

/** The command each assisted entry's skill publishes, keyed by the entry it belongs to. */
const SETUP_COMMANDS: Record<string, string> = {
  "incident-retrospective-drafter": "/incident-retro:setup",
  "github-repo-monitor": "/github-monitor:poll",
};

/**
 * The reference fixtures `OpenHands/extensions` publishes with its catalog.
 * Their request bodies were verified against the live service, and the create
 * model forbids extra keys, so any divergence between the host's derivation
 * and a fixture is a 422 in production rather than a cosmetic difference.
 */
interface FixtureExchange {
  request: { method: string; path: string; body: Record<string, unknown> };
  response: { status: number; body: unknown };
}

interface FixtureScenario {
  id: string;
  formValues?: SetupFormValues;
  localValidation?: { valid: boolean };
  /** Bundle entries only: where the packed archive landed. */
  upload?: { response: { body: { tarball_path: string } } };
  preflight?: FixtureExchange;
  create?: FixtureExchange;
  conversation?: { request: { action: string; message: string } };
  expectedFieldErrors?: Record<string, string>;
  /** False when the recorded request is deliberately not what setup sends. */
  matchesSetupPayload?: boolean;
}

interface FixtureBundle {
  automationId: string;
  scenarios: FixtureScenario[];
}

const BUNDLES = [
  prReviewerFixture,
  repoMonitorFixture,
  incidentFixture,
] as FixtureBundle[];

function requireEntry(automationId: string): SetupEntry {
  const entry = SETUP_REGISTRY.findById(automationId);
  if (!entry) throw new Error(`The registry did not admit ${automationId}`);
  return entry;
}

function requireScenario(
  bundle: FixtureBundle,
  scenarioId: string,
): FixtureScenario {
  const scenario = bundle.scenarios.find(({ id }) => id === scenarioId);
  if (!scenario) {
    throw new Error(`${bundle.automationId} has no scenario ${scenarioId}`);
  }
  return scenario;
}

const CREATE_CASES = BUNDLES.flatMap((bundle) =>
  bundle.scenarios.flatMap((scenario) =>
    scenario.create && scenario.matchesSetupPayload !== false
      ? [
          {
            name: `${bundle.automationId}/${scenario.id}`,
            automationId: bundle.automationId,
            formValues: scenario.formValues ?? {},
            body: scenario.create.request.body,
            // A prompt entry records none; buildCreatePayload ignores it.
            tarballPath: scenario.upload?.response.body.tarball_path,
          },
        ]
      : [],
  ),
);

const PREFLIGHT_CASES = BUNDLES.flatMap((bundle) =>
  bundle.scenarios.flatMap((scenario) =>
    scenario.preflight
      ? [
          {
            name: `${bundle.automationId}/${scenario.id}`,
            automationId: bundle.automationId,
            formValues: scenario.formValues ?? {},
            body: scenario.preflight.request.body,
          },
        ]
      : [],
  ),
);

const CONVERSATION_CASES = BUNDLES.flatMap((bundle) =>
  bundle.scenarios.flatMap((scenario) =>
    scenario.conversation
      ? [
          {
            name: `${bundle.automationId}/${scenario.id}`,
            automationId: bundle.automationId,
            formValues: scenario.formValues ?? {},
            message: scenario.conversation.request.message,
          },
        ]
      : [],
  ),
);

const SERVICE_ERROR_CASES = BUNDLES.flatMap((bundle) =>
  bundle.scenarios.flatMap((scenario) => {
    const exchange = scenario.preflight ?? scenario.create;
    if (!scenario.expectedFieldErrors || !exchange) return [];
    return [
      {
        name: `${bundle.automationId}/${scenario.id}`,
        automationId: bundle.automationId,
        formValues: scenario.formValues ?? {},
        responseBody: exchange.response.body,
        expectedFieldErrors: scenario.expectedFieldErrors,
      },
    ];
  }),
);

describe("the published catalog", () => {
  it.each(BUNDLES.map((bundle) => [bundle.automationId]))(
    "admits %s",
    (automationId) => {
      // Act
      const entry = SETUP_REGISTRY.findById(automationId);

      // Assert
      expect(entry).not.toBeNull();
    },
  );
});

describe("the contract fixtures", () => {
  it("address the endpoints the host calls", () => {
    // Act
    const createPaths = new Set(
      BUNDLES.flatMap((bundle) =>
        bundle.scenarios.flatMap((scenario) =>
          scenario.create ? [scenario.create.request.path] : [],
        ),
      ),
    );
    const preflightPaths = new Set(
      BUNDLES.flatMap((bundle) =>
        bundle.scenarios.flatMap((scenario) =>
          scenario.preflight ? [scenario.preflight.request.path] : [],
        ),
      ),
    );

    // Assert
    // The fixtures cover both creation paths: a prompt entry through the preset
    // endpoint, and a bundle entry through the plain create it uploads to first.
    expect({
      create: [...createPaths].sort(),
      preflight: [...preflightPaths],
    }).toEqual({
      create: [
        automationCreateEndpoint(requireEntry("github-pr-reviewer")),
        automationCreateEndpoint(),
      ].sort(),
      preflight: ["/v1/validate"],
    });
  });
});

describe("buildCreatePayload", () => {
  it.each(CREATE_CASES)(
    "derives the $name create body its fixture pins",
    ({ automationId, formValues, body, tarballPath }) => {
      // Arrange
      const entry = requireEntry(automationId);

      // Act
      const payload = tarballPath
        ? buildCreatePayload(entry, formValues, tarballPath)
        : buildCreatePayload(entry, formValues);

      // Assert
      expect(payload).toEqual(body);
    },
  );

  it("names an automation after the one repository it watches", () => {
    // Arrange
    const entry = requireEntry("github-pr-reviewer");

    // Act
    const payload = buildCreatePayload(entry, {
      repositories: ["OpenHands/automation"],
    });

    // Assert
    expect(payload?.name).toBe(`${entry.name} - OpenHands/automation`);
  });

  it("names an automation watching several through the host's translations", () => {
    // Arrange — several repositories are a count rather than a list of names
    // that would not fit, and a count is a word this host has to translate.
    const { form } = createSetup();
    const entry = createSetupEntry({
      setup: createSetup({
        form: {
          ...form,
          args: {
            ...form.args,
            repository: { ...form.args.repository, multiple: true },
          },
        },
      }),
    });

    // Act
    const payload = buildCreatePayload(entry, {
      repository: ["OpenHands/automation", "OpenHands/extensions"],
      widgetName: "Widgets",
    });

    // Assert
    expect(payload?.name).toBe(`${entry.name} - 2 repositories`);
    expect(translate).toHaveBeenCalledWith("SETUP$REPOSITORY_COUNT", {
      total: 2,
    });
  });

  it("sends no request body for an entry that hands setup to a conversation", () => {
    // Arrange
    const entry = requireEntry("incident-retrospective-drafter");

    // Act
    const payload = buildCreatePayload(entry, {});

    // Assert
    expect(payload).toBeNull();
  });

  it("attaches template provenance when the entry carries a version", () => {
    // Arrange
    const entry = createSetupEntry({ version: "1.0.0" });
    const values = {
      repository: "octo/widgets",
      widgetName: "gadget",
      schedule: "*/5 * * * *",
    };

    // Act
    const payload = buildCreatePayload(entry, values);

    // Assert
    expect(payload?.template).toEqual({
      id: "widget-monitor",
      version: "1.0.0",
      config: values,
    });
  });

  it("sends the payload unchanged for an entry without a version", () => {
    // Arrange
    const entry = createSetupEntry();
    const values = {
      repository: "octo/widgets",
      widgetName: "gadget",
      schedule: "*/5 * * * *",
    };

    // Act
    const payload = buildCreatePayload(entry, values);

    // Assert
    expect(payload).not.toBeNull();
    expect(payload).not.toHaveProperty("template");
  });
});

describe("buildPreflightBody", () => {
  it.each(PREFLIGHT_CASES)(
    "derives the $name preflight envelope its fixture pins",
    ({ automationId, formValues, body }) => {
      // Arrange
      const entry = requireEntry(automationId);

      // Act
      const envelope = buildPreflightBody(entry, formValues);

      // Assert
      expect(envelope).toEqual(body);
    },
  );
});

describe("buildAssistedMessage", () => {
  it.each(CONVERSATION_CASES)(
    "opens $name with the skill command and the fixture's seed message",
    ({ automationId, formValues, message }) => {
      // Arrange
      const entry = requireEntry(automationId);

      // Act
      const seed = buildAssistedMessage(entry, formValues);

      // Assert
      expect(seed).toBe(`${SETUP_COMMANDS[automationId]}\n\n${message}`);
    },
  );
});

describe("service rejections mapped back to fields", () => {
  it.each(SERVICE_ERROR_CASES)(
    "maps the $name rejection to the fields the fixture names",
    ({ automationId, formValues, responseBody, expectedFieldErrors }) => {
      // Arrange
      const entry = requireEntry(automationId);
      const payload = buildCreatePayload(entry, formValues);

      // Act
      const mapped = mapServiceErrors(
        normalizeServiceErrors(responseBody, payload),
        deriveErrorMap(entry),
      );

      // Assert
      expect(mapped).toEqual({
        fieldErrors: expectedFieldErrors,
        formErrors: [],
      });
    },
  );
});

describe("local validation of fixture form values", () => {
  // The unsafe-trigger-phrase case that used to live here is gone: it belonged
  // to github-repo-monitor's event trigger, whose JMESPath filter the phrase was
  // interpolated into. The entry now runs on cron, so no catalog entry declares
  // the `safeExpressionLiteral` constraint any more and there is no fixture to
  // pin. The constraint itself is still exercised, on a synthetic setup, by
  // `manifest-local-validation.test.ts`.

  it("passes an entirely blank assisted form, as its fixture records", () => {
    // Arrange
    const scenario = requireScenario(BUNDLES[2], "nothing-filled-in");
    const entry = requireEntry("incident-retrospective-drafter");

    // Act
    const errors = validateFormValues(entry.setup, scenario.formValues ?? {});

    // Assert
    expect(errors).toEqual({});
  });
});

describe("deriveErrorMap", () => {
  it("recovers which fields built each payload path", () => {
    // Act
    const errorMap = deriveErrorMap(requireEntry("github-pr-reviewer"));

    // Assert — a bundle's answers reach the service through its rendered
    // config rather than through a prompt, so the paths are the config's.
    expect(errorMap).toEqual({
      name: ["repositories"],
      "trigger.schedule": ["schedule"],
      "trigger.timezone": ["timezone"],
      "template.config.repos": ["repositories"],
      "template.config.trigger_label": ["triggerLabel"],
      "template.config.review_tone": ["reviewTone"],
    });
  });
});
