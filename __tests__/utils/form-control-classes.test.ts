import { describe, expect, it } from "vitest";
import {
  formControlButtonClassName,
  formControlFieldClassName,
  formControlShellClassName,
} from "#/utils/form-control-classes";
import { dropdownFilterTriggerClassName } from "#/utils/dropdown-classes";

describe("formControlClasses", () => {
  it("standardizes fields, shells, and buttons to 36px with rounded-lg", () => {
    expect(formControlFieldClassName).toContain("h-9");
    expect(formControlFieldClassName).toContain("rounded-lg");
    expect(formControlFieldClassName).toContain("border-border-input");
    expect(formControlFieldClassName).toContain("dark:border-border");
    expect(formControlFieldClassName).toContain("bg-base-secondary");

    expect(formControlShellClassName).toContain("h-9");
    expect(formControlShellClassName).toContain("rounded-lg");
    expect(formControlShellClassName).toContain("focus-within:ring-1");

    expect(formControlButtonClassName).toContain("h-9");
    expect(formControlButtonClassName).toContain("rounded-lg");
  });

  it("preserves dark focus borders for fields, shells, and filter triggers", () => {
    expect(formControlFieldClassName.split(/\s+/)).toContain(
      "dark:focus:border-contrast/40",
    );
    expect(formControlShellClassName.split(/\s+/)).toContain(
      "dark:focus-within:border-contrast/40",
    );
    expect(dropdownFilterTriggerClassName.split(/\s+/)).toContain(
      "dark:focus-visible:border-contrast/40",
    );
  });
});
