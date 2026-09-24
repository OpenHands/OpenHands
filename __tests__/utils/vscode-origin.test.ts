import { afterEach, describe, expect, it } from "vitest";
import {
  getOriginVSCodeBasePath,
  isVSCodeUrlServedByOrigin,
} from "#/utils/vscode-origin";

function setInjected(value: unknown) {
  if (value === undefined) {
    delete (window as unknown as Record<string, unknown>)
      .__AGENT_CANVAS_VSCODE_BASE_PATH__;
    return;
  }
  (
    window as unknown as Record<string, unknown>
  ).__AGENT_CANVAS_VSCODE_BASE_PATH__ = value;
}

afterEach(() => setInjected(undefined));

describe("getOriginVSCodeBasePath", () => {
  it("returns null when nothing advertises an editor", () => {
    // The public-mode origin and any deployment predating this flag land here.
    // Hiding the control matches the behavior every local backend had before
    // the control was made to render at all, so this default cannot regress a
    // deployment that used to work.
    expect(getOriginVSCodeBasePath()).toBeNull();
  });

  it("reads the value static-server injects into the document", () => {
    setInjected("/vscode");
    expect(getOriginVSCodeBasePath()).toBe("/vscode");
  });

  it("normalizes a missing leading slash and a trailing slash", () => {
    // The flag is validated at the server, but the value also arrives from
    // VITE_VSCODE_BASE_PATH, so normalize rather than trust the shape.
    setInjected("vscode/");
    expect(getOriginVSCodeBasePath()).toBe("/vscode");
  });

  it("treats a blank or non-string value as no editor", () => {
    setInjected("   ");
    expect(getOriginVSCodeBasePath()).toBeNull();
    setInjected(42);
    expect(getOriginVSCodeBasePath()).toBeNull();
  });
});

describe("isVSCodeUrlServedByOrigin", () => {
  const origin = window.location.origin;

  it("accepts a URL under the advertised prefix", () => {
    expect(
      isVSCodeUrlServedByOrigin(`${origin}/vscode/?tkn=abc`, "/vscode"),
    ).toBe(true);
  });

  it("accepts the prefix itself with no trailing path", () => {
    expect(isVSCodeUrlServedByOrigin(`${origin}/vscode`, "/vscode")).toBe(true);
  });

  it("rejects the origin root, which is what an unprefixed backend returns", () => {
    // The extra-backend case: agent-server appends nothing, so the "editor"
    // URL is the canvas itself. Same origin, wrong destination.
    expect(isVSCodeUrlServedByOrigin(`${origin}/?tkn=abc`, "/vscode")).toBe(
      false,
    );
  });

  it("rejects a prefix that merely shares a string prefix", () => {
    // `/vscode-other` starts with `/vscode` as a string but is a different
    // route; only a full segment boundary counts.
    expect(
      isVSCodeUrlServedByOrigin(`${origin}/vscode-other/`, "/vscode"),
    ).toBe(false);
  });

  it("rejects a cross-origin URL", () => {
    expect(
      isVSCodeUrlServedByOrigin("http://elsewhere.test/vscode/", "/vscode"),
    ).toBe(false);
  });

  it("rejects when there is no advertised prefix or no URL", () => {
    expect(isVSCodeUrlServedByOrigin(`${origin}/vscode/`, null)).toBe(false);
    expect(isVSCodeUrlServedByOrigin(null, "/vscode")).toBe(false);
    expect(isVSCodeUrlServedByOrigin("not a url", "/vscode")).toBe(false);
  });
});
