import { describe, expect, it, vi } from "vitest";
import incidentFixture from "@openhands/extensions/testing/automations/incident-retrospective-drafter.json";
import prReviewerFixture from "@openhands/extensions/testing/automations/github-pr-reviewer.json";
import repoMonitorFixture from "@openhands/extensions/testing/automations/github-repo-monitor.json";
import {
  AUTOMATION_CREATE_ENDPOINT,
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
  ],
}));

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
    expect({ create: [...createPaths], preflight: [...preflightPaths] }).toEqual(
      { create: [AUTOMATION_CREATE_ENDPOINT], preflight: ["/v1/validate"] },
    );
  });
});

describe("buildCreatePayload", () => {
  it.each(CREATE_CASES)(
    "derives the $name create body its fixture pins",
    ({ automationId, formValues, body }) => {
      // Arrange
      const entry = requireEntry(automationId);

      // Act
      const payload = buildCreatePayload(entry, formValues);

      // Assert
      expect(payload).toEqual(body);
    },
  );

  it("sends no request body for an entry that hands setup to a conversation", () => {
    // Arrange
    const entry = requireEntry("incident-retrospective-drafter");

    // Act
    const payload = buildCreatePayload(entry, {});

    // Assert
    expect(payload).toBeNull();
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
      expect(seed).toBe(`/incident-retro:setup\n\n${message}`);
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
      expect(mapped).toEqual({ fieldErrors: expectedFieldErrors, formErrors: [] });
    },
  );
});

describe("local validation of fixture form values", () => {
  it("blocks the unsafe trigger phrase before any request is made", () => {
    // Arrange — the fixture names the failing field; the code is the host's
    // own vocabulary, rendered through its translations.
    const scenario = requireScenario(
      BUNDLES[1],
      "quote-in-trigger-phrase-blocked-locally",
    );
    const entry = requireEntry("github-repo-monitor");

    // Act
    const errors = validateFormValues(entry.setup, scenario.formValues ?? {});

    // Assert
    expect(errors).toEqual({
      triggerPhrase: { code: "unsafeExpressionLiteral" },
    });
  });

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
