import { describe, expect, it } from "vitest";
import { isCrossAppPath } from "./cross-app-redirect";

describe("isCrossAppPath", () => {
  it("detects automation app routes", () => {
    expect(isCrossAppPath("/automations")).toBe(true);
    expect(isCrossAppPath("/automations?login_method=github")).toBe(true);
    expect(isCrossAppPath("/automations/abc?tab=runs")).toBe(true);
  });

  it("detects Canvas app routes", () => {
    expect(isCrossAppPath("/canvas")).toBe(true);
    expect(isCrossAppPath("/canvas?login_method=github")).toBe(true);
    expect(isCrossAppPath("/canvas/conversations/abc")).toBe(true);
  });

  it("does not match similarly-prefixed main app routes", () => {
    expect(isCrossAppPath("/automation")).toBe(false);
    expect(isCrossAppPath("/automations-old")).toBe(false);
    expect(isCrossAppPath("/canvases")).toBe(false);
    expect(isCrossAppPath("/settings?returnTo=/automations")).toBe(false);
    expect(isCrossAppPath("/settings?returnTo=/canvas")).toBe(false);
  });

  it("does not treat off-origin targets as cross-app paths", () => {
    expect(isCrossAppPath("//evil.com/automations")).toBe(false);
    expect(isCrossAppPath("https://evil.com/automations")).toBe(false);
    expect(isCrossAppPath("//evil.com/canvas")).toBe(false);
    expect(isCrossAppPath("https://evil.com/canvas")).toBe(false);
  });
});
