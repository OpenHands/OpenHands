import { afterEach, describe, expect, it, vi } from "vitest";
import {
  cn,
  downloadBlob,
  getFileExtension,
  getStyleHeightPx,
  isMobileDevice,
  isMobileUserAgent,
  setStyleHeightPx,
} from "#/utils/utils";

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

const useDesktopNavigator = (maxTouchPoints = 0) => {
  const originalNavigator = navigator;
  vi.stubGlobal(
    "navigator",
    new Proxy(originalNavigator, {
      get(target, property, receiver) {
        if (property === "userAgent") {
          return "Mozilla/5.0 (X11; Linux x86_64)";
        }
        if (property === "maxTouchPoints") return maxTouchPoints;
        return Reflect.get(target, property, receiver);
      },
    }),
  );
};

const useWindowCapabilities = ({
  hasTouchProperty,
  finePointer,
}: {
  hasTouchProperty: boolean;
  finePointer?: boolean;
}) => {
  const originalWindow = window;
  const windowProxy = new Proxy(originalWindow, {
    has(target, property) {
      if (property === "ontouchstart") return hasTouchProperty;
      return Reflect.has(target, property);
    },
    get(target, property, receiver) {
      if (property === "matchMedia") {
        if (finePointer === undefined) return undefined;
        return vi.fn((query: string) => ({
          matches: query === "(pointer: fine)" ? finePointer : !finePointer,
        }));
      }
      return Reflect.get(target, property, receiver);
    },
  });

  vi.stubGlobal("window", windowProxy);
};

describe("class name and element sizing behavior", () => {
  it("combines conditional classes and lets later Tailwind utilities win", () => {
    expect(cn("px-2 text-sm", { hidden: true }, "px-6")).toBe(
      "text-sm hidden px-6",
    );
  });

  it("reads numeric pixel heights and falls back for an unset height", () => {
    const element = document.createElement("div");

    element.style.height = "42.5px";
    expect(getStyleHeightPx(element, 12)).toBe(42.5);

    element.style.height = "";
    expect(getStyleHeightPx(element, 12)).toBe(12);
  });

  it("sets an element height in pixels", () => {
    const element = document.createElement("textarea");

    setStyleHeightPx(element, 128);

    expect(element.style.height).toBe("128px");
  });
});

describe("browser download behavior", () => {
  it("clicks a temporary named link and releases its object URL", () => {
    const anchor = document.createElement("a");
    const originalCreateElement = document.createElement.bind(document);
    const click = vi.spyOn(anchor, "click").mockImplementation(() => {});
    vi.spyOn(document, "createElement").mockImplementation(
      (tagName, options) =>
        tagName === "a" ? anchor : originalCreateElement(tagName, options),
    );
    const createObjectURL = vi.fn().mockReturnValue("blob:report");
    const revokeObjectURL = vi.fn();
    const originalWindow = window;
    vi.stubGlobal(
      "window",
      new Proxy(originalWindow, {
        get(target, property, receiver) {
          if (property === "URL") {
            return { createObjectURL, revokeObjectURL };
          }
          return Reflect.get(target, property, receiver);
        },
      }),
    );
    const blob = new Blob(["coverage"], { type: "text/plain" });

    downloadBlob(blob, "coverage.txt");

    expect(createObjectURL).toHaveBeenCalledWith(blob);
    expect(anchor.href).toBe("blob:report");
    expect(anchor.download).toBe("coverage.txt");
    expect(click).toHaveBeenCalledOnce();
    expect(anchor.isConnected).toBe(false);
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:report");
  });
});

describe("mobile environment detection", () => {
  it("recognizes a mobile user agent independently of touch support", () => {
    vi.spyOn(window.navigator, "userAgent", "get").mockReturnValue(
      "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)",
    );
    useWindowCapabilities({ hasTouchProperty: false, finePointer: false });

    expect(isMobileUserAgent()).toBe(true);
    expect(isMobileDevice()).toBe(true);
  });

  it("does not classify a desktop without touch input as mobile", () => {
    useDesktopNavigator();
    useWindowCapabilities({ hasTouchProperty: false, finePointer: false });

    expect(isMobileUserAgent()).toBe(false);
    expect(isMobileDevice()).toBe(false);
  });

  it("does not classify a touchscreen laptop with a fine pointer as mobile", () => {
    useDesktopNavigator(2);
    useWindowCapabilities({ hasTouchProperty: false, finePointer: true });

    expect(isMobileDevice()).toBe(false);
  });

  it("classifies a touch device with a coarse primary pointer as mobile", () => {
    useDesktopNavigator();
    useWindowCapabilities({ hasTouchProperty: true, finePointer: false });

    expect(isMobileDevice()).toBe(true);
  });

  it("assumes a touch device is mobile when pointer media queries are unavailable", () => {
    useDesktopNavigator(1);
    useWindowCapabilities({ hasTouchProperty: false });

    expect(isMobileDevice()).toBe(true);
  });
});

describe("file type labels", () => {
  it.each([
    ["archive.tar.gz", "GZ"],
    ["photo.Jpeg", "JPEG"],
    ["README", "README"],
    ["untitled.", "FILE"],
    ["", "FILE"],
  ])("labels %j as %s", (filename, expected) => {
    expect(getFileExtension(filename)).toBe(expected);
  });
});
