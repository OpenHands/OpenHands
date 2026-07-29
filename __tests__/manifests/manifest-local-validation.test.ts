import { describe, expect, it } from "vitest";
import {
  getFieldOptions,
  getInitialFormValues,
  resolveFieldOverrides,
  validateFormValues,
} from "#/manifests/manifest-local-validation";
import type { DeploymentCapabilities } from "#/manifests/types";
import { createSetup } from "./manifest-test-data";

const SETUP = createSetup({
  form: {
    args: {
      widgetName: {
        type: "text",
        label: "Widget name",
        help: "What to call it.",
        required: true,
        constraints: { maxLength: 10 },
      },
      matchPhrase: {
        type: "text",
        label: "Match phrase",
        help: "Becomes part of a filter expression.",
        required: false,
        constraints: { format: "safeExpressionLiteral" },
      },
      size: {
        type: "select",
        label: "Size",
        help: "How big.",
        default: "small",
        required: false,
        options: [{ value: "small", label: "Small" }],
      },
      timezone: {
        type: "timezone",
        label: "Timezone",
        help: "Which zone the schedule is read in.",
        required: false,
      },
    },
  },
});

describe("validateFormValues", () => {
  it("reports a required field the user left blank", () => {
    // Act
    const errors = validateFormValues(SETUP, { widgetName: "   " });

    // Assert
    expect(errors.widgetName).toEqual({ code: "required" });
  });

  it("reports a value longer than the manifest allows", () => {
    // Act
    const errors = validateFormValues(SETUP, {
      widgetName: "a-very-long-widget-name",
    });

    // Assert
    expect(errors.widgetName).toEqual({ code: "maxLength", length: 10 });
  });

  it("rejects characters that would break out of an expression literal", () => {
    // Act
    const errors = validateFormValues(SETUP, {
      widgetName: "ok",
      matchPhrase: 'he said "hi"',
    });

    // Assert
    expect(errors.matchPhrase).toEqual({ code: "unsafeExpressionLiteral" });
  });

  it("rejects a choice that is not on offer", () => {
    // Act
    const errors = validateFormValues(SETUP, {
      widgetName: "ok",
      size: "enormous",
    });

    // Assert
    expect(errors.size).toEqual({ code: "invalidOption" });
  });

  it("passes a form that satisfies every declared constraint", () => {
    // Act
    const errors = validateFormValues(SETUP, {
      widgetName: "ok",
      matchPhrase: "ship it",
      size: "small",
    });

    // Assert
    expect(errors).toEqual({});
  });
});

describe("deployment-supplied constraints", () => {
  it("offers the timezones the deployment reports, which no manifest declares", () => {
    // Arrange — the accepted zones belong to the deployment, not the manifest.
    const capabilities = {
      triggers: { cron: { minIntervalSeconds: 60, timezones: ["UTC", "CET"] } },
    } as DeploymentCapabilities;

    // Act
    const overrides = resolveFieldOverrides(SETUP, capabilities);

    // Assert
    expect(
      getFieldOptions("timezone", SETUP.form.args.timezone, overrides),
    ).toEqual([
      { value: "UTC", label: "UTC" },
      { value: "CET", label: "CET" },
    ]);
  });

  it("keeps the manifest's own options when the deployment cannot be asked", () => {
    // Act
    const overrides = resolveFieldOverrides(SETUP, null);

    // Assert
    expect(getFieldOptions("size", SETUP.form.args.size, overrides)).toEqual(
      SETUP.form.args.size.options,
    );
  });
});

describe("getInitialFormValues", () => {
  it("seeds every declared field with its declared default", () => {
    // Act
    const values = getInitialFormValues(SETUP);

    // Assert
    expect(values).toEqual({
      widgetName: "",
      matchPhrase: "",
      size: "small",
      timezone: "",
    });
  });
});
