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
});
