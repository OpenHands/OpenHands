import { AUTOMATION_CATALOG } from "@openhands/extensions/automations";
import customAutomationFixture from "@openhands/extensions/testing/automations/custom-automation.json";
import { describe, expect, it } from "vitest";
import { buildCreatePayload, buildPreflightBody } from "./automation-setup";
import type { SetupEntry, SetupFormValues, SetupRequestBody } from "./types";
import { validateSetupEntry } from "./manifest-validation";

const customAutomation = AUTOMATION_CATALOG.find(
  (entry) => entry.id === "custom-automation",
) as SetupEntry | undefined;

describe("custom automation setup actions", () => {
  it("uses the selected agent profile instead of a second model authority", () => {
    const scenario = customAutomationFixture.scenarios[0];
    const values = {
      ...scenario.formValues,
      model: "other-model",
      agent_profile_id: "11111111-1111-4111-8111-111111111111",
    } as SetupFormValues;
    const payload = buildCreatePayload(
      customAutomation!,
      values,
      undefined,
      scenario.selectedTrigger,
      scenario.selectedAction,
    );
    expect(payload?.agent_profile_id).toBe(values.agent_profile_id);
    expect(payload).not.toHaveProperty("model");
  });
  it("admits the custom automation action manifest", () => {
    expect(customAutomation).toBeDefined();
    expect(validateSetupEntry(customAutomation).errors).toEqual([]);
  });

  it.each(customAutomationFixture.scenarios)(
    "derives the $id preflight and create payloads",
    (scenario) => {
      expect(customAutomation).toBeDefined();
      const formValues = scenario.formValues as unknown as SetupFormValues;
      const createBody = scenario.create.request
        .body as unknown as SetupRequestBody;
      const tarballPath =
        typeof createBody.tarball_path === "string"
          ? createBody.tarball_path
          : undefined;

      expect(
        buildPreflightBody(
          customAutomation!,
          formValues,
          scenario.selectedTrigger,
          scenario.selectedAction,
        ),
      ).toEqual(scenario.preflight.request.body);
      expect(
        buildCreatePayload(
          customAutomation!,
          formValues,
          tarballPath,
          scenario.selectedTrigger,
          scenario.selectedAction,
        ),
      ).toEqual(createBody);
    },
  );

  it("keeps event filters rendered from non-filter form fields", () => {
    const entry: SetupEntry = {
      id: "mention-responder",
      name: "Mention responder",
      description: "Respond to matching comments.",
      requires: { integrations: {} },
      setup: {
        version: "1.0",
        mode: "direct",
        form: {
          triggers: {
            event: {
              source: {
                type: "event-source",
                label: "Source",
                help: "Where events come from.",
                required: true,
              },
              on: {
                type: "event-type",
                label: "Event",
                help: "Which event to watch.",
                required: true,
              },
              mention: {
                type: "text",
                label: "Mention",
                help: "Text that must appear in the comment.",
                required: true,
              },
            },
          },
          args: {},
        },
        prompt: "Handle {{form.mention}}.",
        filter: "icontains(comment.body, '{{form.mention}}')",
      },
    };

    expect(
      buildCreatePayload(
        entry,
        {
          source: "github",
          on: "issue_comment.created",
          mention: "@openhands",
        },
        undefined,
        "event",
      ),
    ).toEqual({
      name: "Mention responder",
      prompt: "Handle @openhands.",
      trigger: {
        type: "event",
        source: "github",
        on: "issue_comment.created",
        filter: "icontains(comment.body, '@openhands')",
      },
    });
  });
});
