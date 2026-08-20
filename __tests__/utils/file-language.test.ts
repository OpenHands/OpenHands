import { describe, it, expect } from "vitest";

import { getPrismLanguageForFile } from "#/utils/file-language";

describe("getPrismLanguageForFile", () => {
  it("maps common source-code extensions to their Prism grammars", () => {
    expect(getPrismLanguageForFile("src/index.ts")).toBe("typescript");
    expect(getPrismLanguageForFile("src/App.tsx")).toBe("tsx");
    expect(getPrismLanguageForFile("scripts/build.js")).toBe("javascript");
    expect(getPrismLanguageForFile("main.py")).toBe("python");
    expect(getPrismLanguageForFile("server.go")).toBe("go");
    expect(getPrismLanguageForFile("lib/util.rs")).toBe("rust");
    expect(getPrismLanguageForFile("Cargo.toml")).toBe("toml");
  });

  it("maps web / markup files to grammars", () => {
    expect(getPrismLanguageForFile("index.html")).toBe("markup");
    expect(getPrismLanguageForFile("page.htm")).toBe("markup");
    expect(getPrismLanguageForFile("logo.svg")).toBe("markup");
    expect(getPrismLanguageForFile("styles/main.css")).toBe("css");
    expect(getPrismLanguageForFile("styles/main.scss")).toBe("scss");
    expect(getPrismLanguageForFile("README.md")).toBe("markdown");
    expect(getPrismLanguageForFile("docs/guide.mdx")).toBe("markdown");
  });

  it("recognizes well-known no-extension filenames", () => {
    expect(getPrismLanguageForFile("Dockerfile")).toBe("docker");
    expect(getPrismLanguageForFile("path/to/Dockerfile")).toBe("docker");
    expect(getPrismLanguageForFile("Makefile")).toBe("makefile");
    expect(getPrismLanguageForFile(".bashrc")).toBe("bash");
    expect(getPrismLanguageForFile(".gitignore")).toBe("bash");
  });

  it("is case-insensitive on the extension", () => {
    expect(getPrismLanguageForFile("README.MD")).toBe("markdown");
    expect(getPrismLanguageForFile("Util.PY")).toBe("python");
  });

  it("falls back on mime type when no extension matches", () => {
    expect(
      getPrismLanguageForFile("LICENSE", "text/markdown"),
    ).toBe("markdown");
    expect(
      getPrismLanguageForFile("data", "application/json"),
    ).toBe("json");
    expect(
      getPrismLanguageForFile("config", "text/yaml"),
    ).toBe("yaml");
  });

  it("returns null for unknown extensions and unknown mime types", () => {
    // No extension, no mime type, no known basename → bail out so the
    // caller can render a raw <pre>.
    expect(getPrismLanguageForFile("LICENSE")).toBeNull();
    expect(getPrismLanguageForFile("data.xyz")).toBeNull();
    expect(
      getPrismLanguageForFile("data.bin", "application/octet-stream"),
    ).toBeNull();
  });

  it("does not confuse a dot-prefix file with an extension", () => {
    // `.env` is a basename, not "extension = env".
    expect(getPrismLanguageForFile(".env")).toBe("bash");
    expect(getPrismLanguageForFile(".dockerignore")).toBe("bash");
  });
});
