import { describe, expect, it } from "vitest";
import {
  assertCanvasExtensionModule,
  InvalidCanvasExtensionModuleError,
} from "#/extensions/canvas-extension-module-loader";

describe("assertCanvasExtensionModule", () => {
  it("accepts a module namespace with an activate export", () => {
    expect(() =>
      assertCanvasExtensionModule({ activate: () => undefined }),
    ).not.toThrow();
  });

  it("rejects modules that do not implement the host lifecycle", () => {
    expect(() => assertCanvasExtensionModule({ default: {} })).toThrow(
      InvalidCanvasExtensionModuleError,
    );
  });
});
