import { describe, expect, it } from "vitest";
import {
  getFieldOptions,
  getInitialFormValues,
  resolveFieldOverrides,
  validateFormValues,
} from "#/manifests/manifest-local-validation";
import { createManifest } from "./manifest-test-data";

const FIELDS = [
  {
    name: "widgetName",
    type: "text" as const,
    label: "Widget name",
    help: "What to call it.",
    required: true,
    constraints: { maxLength: 10 },
  },
  {
    name: "matchPhrase",
    type: "text" as const,
    label: "Match phrase",
    help: "Becomes part of a filter expression.",
    required: false,
    constraints: { format: "safeExpressionLiteral" as const },
  },
  {
    name: "size",
    type: "select" as const,
    label: "Size",
    help: "How big.",
    required: false,
    options: [{ value: "small", label: "Small" }],
  },
];

const manifest = createManifest({ form: { fields: FIELDS } });

describe("validateFormValues", () => {
  it("reports a required field the user left blank", () => {
    // Act
    const errors = validateFormValues(manifest, { widgetName: "   " });

    // Assert
    expect(errors.widgetName).toEqual({ code: "required" });
  });

  it("reports a value longer than the manifest allows", () => {
    // Act
    const errors = validateFormValues(manifest, {
      widgetName: "a-very-long-widget-name",
    });

    // Assert
    expect(errors.widgetName).toEqual({ code: "maxLength", length: 10 });
  });

  it("rejects characters that would break out of an expression literal", () => {
    // Act
    const errors = validateFormValues(manifest, {
      widgetName: "ok",
      matchPhrase: 'he said "hi"',
    });

    // Assert
    expect(errors.matchPhrase).toEqual({ code: "unsafeExpressionLiteral" });
  });

  it("rejects a choice that is not on offer", () => {
    // Act
    const errors = validateFormValues(manifest, {
      widgetName: "ok",
      size: "enormous",
    });

    // Assert
    expect(errors.size).toEqual({ code: "invalidOption" });
  });

  it("passes a form that satisfies every declared constraint", () => {
    // Act
    const errors = validateFormValues(manifest, {
      widgetName: "ok",
      matchPhrase: "ship it",
      size: "small",
    });

    // Assert
    expect(errors).toEqual({});
  });
});

describe("deployment-supplied constraints", () => {
  it("offers the options the deployment reports instead of the declared ones", () => {
    // Arrange — the deployment, not the manifest, knows which regions exist.
    const bound = createManifest({
      form: {
        fields: [
          {
            name: "region",
            type: "select",
            label: "Region",
            help: "Where it runs.",
            required: true,
          },
        ],
      },
      capabilities: {
        discovery: { method: "GET", path: "/v1/capabilities" },
        requires: { ready: true },
        bindings: [
          { field: "region", constraint: "options", from: "regions.available" },
        ],
        onUnsupported: { behavior: "block", message: "Unavailable here." },
      },
    });

    // Act
    const overrides = resolveFieldOverrides(bound, {
      regions: { available: ["eu-west", "us-east"] },
    });

    // Assert
    expect(getFieldOptions(bound.form.fields[0], overrides)).toEqual([
      { value: "eu-west", label: "eu-west" },
      { value: "us-east", label: "us-east" },
    ]);
  });

  it("keeps the manifest's own options when the deployment cannot be asked", () => {
    // Act
    const overrides = resolveFieldOverrides(manifest, null);

    // Assert
    expect(getFieldOptions(FIELDS[2], overrides)).toEqual(FIELDS[2].options);
  });
});

describe("getInitialFormValues", () => {
  it("seeds every declared field with its declared default", () => {
    // Act
    const values = getInitialFormValues(manifest);

    // Assert
    expect(values).toEqual({ widgetName: "", matchPhrase: "", size: "" });
  });
});
