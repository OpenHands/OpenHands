import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

const SRC = path.resolve(__dirname, "../../src");

function sourceFiles(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) return sourceFiles(full);
    return /\.tsx?$/.test(entry.name) && !/\.test\.tsx?$/.test(entry.name)
      ? [full]
      : [];
  });
}

const SOURCES = sourceFiles(SRC).map((file) => ({
  file: path.relative(SRC, file),
  text: readFileSync(file, "utf8"),
}));

function offenders(pattern: RegExp, allow: string[] = []): string[] {
  return SOURCES.filter(
    ({ file, text }) => !allow.includes(file) && pattern.test(text),
  ).map(({ file }) => file);
}

describe("theme source guards", () => {
  it("routes portals through AppearancePortal so dark: utilities keep matching", () => {
    expect(
      offenders(/createPortal\(/, ["components/shared/appearance-portal.tsx"]),
    ).toEqual([]);
  });

  it("does not hard-code white icon ink that vanishes on light surfaces", () => {
    expect(offenders(/\bcolor=(?:\{\s*)?"(?:white|#fff|#ffffff)"/i)).toEqual(
      [],
    );
  });

  it("keeps dark-palette literals behind the dark: variant", () => {
    const literal =
      /(?:^|[\s"'`])(?:[a-z-]+:)*(?:text|bg|border)-\[(?:#717888|#A3A3A3|#3D4046|#F87171|#fff|rgba\(71,\s*74,\s*84,[^\]]*\))\]/i;
    const unguarded = SOURCES.flatMap(({ file, text }) =>
      text
        .split(/(?=[\s"'`])/)
        .filter((token) => literal.test(token) && !/\bdark:/.test(token))
        .map((token) => `${file}: ${token.trim()}`),
    );
    expect(unguarded).toEqual([]);
  });

  it("paints the shared close glyph with currentColor", () => {
    const svg = readFileSync(path.join(SRC, "icons/close.svg"), "utf8");
    expect(svg).toContain('fill="currentColor"');
    expect(svg).not.toMatch(/fill="(?:white|#fff)"/i);
  });

  // close.svg used to hard-code fill="white"; every render must keep that
  // literal white in dark themes, where --oh-contrast is consumer-overridable.
  it("keeps every close.svg render literal white in dark themes", () => {
    const importer =
      /import\s+(\w+)\s+from\s+["'][^"']*icons\/close\.svg(?:\?react)?["']/;
    const sites = SOURCES.flatMap(({ file, text }) => {
      const name = text.match(importer)?.[1];
      if (!name) return [];
      return [...text.matchAll(new RegExp(`<${name}\\b[^>]*>`, "g"))].map(
        ([tag]) => ({ file, tag }),
      );
    });

    expect(sites.length).toBeGreaterThanOrEqual(5);
    const unpinned = sites
      .filter(({ tag }) => {
        const classes = tag.match(/className="([^"]*)"/)?.[1].split(/\s+/);
        return !classes?.some(
          (c) => c === "text-white" || c === "dark:text-white",
        );
      })
      .map(({ file, tag }) => `${file}: ${tag.replace(/\s+/g, " ")}`);
    expect(unpinned).toEqual([]);
  });
});
