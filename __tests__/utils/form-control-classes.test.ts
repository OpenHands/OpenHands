import { describe, expect, it } from "vitest";
import {
  formControlButtonClassName,
  formControlFieldClassName,
  formControlShellClassName,
} from "#/utils/form-control-classes";

describe("formControlClasses", () => {
  it("standardizes fields, shells, and buttons to 36px with rounded-lg", () => {
    expect(formControlFieldClassName).toContain("h-9");
    expect(formControlFieldClassName).toContain("rounded-lg");
    expect(formControlFieldClassName).toContain("border-[var(--oh-border)]");
    expect(formControlFieldClassName).toContain("bg-base-secondary");

    expect(formControlShellClassName).toContain("h-9");
    expect(formControlShellClassName).toContain("rounded-lg");
    expect(formControlShellClassName).toContain("focus-within:ring-1");

    expect(formControlButtonClassName).toContain("h-9");
    expect(formControlButtonClassName).toContain("rounded-lg");
  });
});
