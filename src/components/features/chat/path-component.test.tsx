/* eslint-disable i18next/no-literal-string -- test fixtures use literal path strings */
import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import EventLogger from "#/utils/event-logger";
import { PathComponent, isLikelyDirectory } from "./path-component";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("isLikelyDirectory", () => {
  it.each([
    ["", false],
    ["src/", true],
    ["src\\", true],
    ["src/components", true],
    ["src/components/file.tsx", false],
    ["\\file.txt", false],
    [".gitignore", false],
  ])("classifies %j as %s", (path, expected) => {
    expect(isLikelyDirectory(path)).toBe(expected);
  });
});

describe("PathComponent", () => {
  it("shows a decoded Unix filename and preserves the decoded path as its title", () => {
    render(<PathComponent>/workspace/a&amp;b/readme.md</PathComponent>);

    expect(screen.getByText("readme.md")).toHaveAttribute(
      "title",
      "/workspace/a&b/readme.md",
    );
  });

  it("extracts filenames from Windows paths", () => {
    render(<PathComponent>{"C:\\workspace\\src\\index.ts"}</PathComponent>);

    expect(screen.getByText("index.ts")).toHaveAttribute(
      "title",
      "C:\\workspace\\src\\index.ts",
    );
  });

  it("marks an extensionless final segment as a directory", () => {
    render(<PathComponent>/workspace/project/docs</PathComponent>);

    expect(screen.getByText("docs/")).toHaveAttribute(
      "title",
      "/workspace/project/docs",
    );
  });

  it("renders an empty string as an empty path", () => {
    const { container } = render(<PathComponent>{""}</PathComponent>);

    const pathElement = container.querySelector(
      '[data-testid="path-component-link"]',
    );
    expect(pathElement).toHaveAttribute("title", "");
    expect(pathElement).toBeEmptyDOMElement();
  });

  it("processes string array entries without replacing React nodes", () => {
    render(
      <PathComponent>
        {["/workspace/src/app.tsx", <em key="separator">through</em>, "tests"]}
      </PathComponent>,
    );

    expect(screen.getByText("app.tsx")).toHaveAttribute(
      "title",
      "/workspace/src/app.tsx",
    );
    expect(screen.getByText("through").tagName).toBe("EM");
    expect(screen.getByText("tests/")).toHaveAttribute("title", "tests");
    expect(screen.getByText("through").closest("span")).toHaveClass(
      "font-normal",
    );
  });

  it("renders multiple string paths without React key collisions", () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    render(
      <PathComponent>
        {["/workspace/src/one.ts", "/workspace/src/two.ts"]}
      </PathComponent>,
    );

    expect(screen.getByText("one.ts")).toBeInTheDocument();
    expect(screen.getByText("two.ts")).toBeInTheDocument();
    expect(errorSpy).not.toHaveBeenCalled();
  });

  it("passes through a single non-string child", () => {
    render(
      <PathComponent>
        <em>unchanged</em>
      </PathComponent>,
    );

    expect(screen.getByText("unchanged").tagName).toBe("EM");
    expect(screen.getByText("unchanged").closest("span")).toHaveClass(
      "font-mono",
    );
  });

  it("falls back to the original path and logs decoding errors", () => {
    const createElement = document.createElement.bind(document);
    vi.spyOn(document, "createElement").mockImplementation(
      (tagName, options) => {
        if (tagName === "textarea") {
          throw new Error("decode failed");
        }
        return createElement(tagName, options);
      },
    );
    const errorSpy = vi
      .spyOn(EventLogger, "error")
      .mockImplementation(() => {});

    render(<PathComponent>/workspace/readme.md</PathComponent>);

    expect(screen.getByText("/workspace/readme.md")).not.toHaveAttribute(
      "title",
    );
    expect(errorSpy).toHaveBeenCalledWith("Error: decode failed");
  });
});
