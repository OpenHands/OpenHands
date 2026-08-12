import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

/**
 * The mock-LLM specs run serially (`workers: 1`) under a single
 * `globalTimeout`, so every per-test ceiling is drawn from one shared budget.
 * A test allowed a large share of that budget turns a single stall into a
 * whole-suite failure: the run is killed at the cap and the remaining tests
 * never report, so the result says nothing about the change under test.
 *
 * These tests pin the relationship between the two numbers rather than the
 * numbers themselves, so the suite can be retuned without editing this file —
 * but not in a way that reintroduces the imbalance. Both configs that run
 * these specs are checked, since a ceiling is shared by all of them.
 */

const SPEC_ROOT = join(process.cwd(), "tests/e2e/mock-llm");

const CONFIGS = [
  "playwright.mock-llm.config.ts",
  "playwright.mock-llm-docker.config.ts",
] as const;

/** No single test may claim more than this share of the smallest budget. */
const MAX_SHARE_OF_GLOBAL_BUDGET = 0.25;

const toMs = (literal: string): number => Number(literal.replace(/_/g, ""));

/**
 * Read the CI branch of `globalTimeout`. The docker config routes it through a
 * named constant that falls back to a default, so resolve one hop through the
 * file's numeric `const` declarations rather than grabbing the next number.
 */
function readGlobalTimeoutMs(configFile: string): number {
  const source = readFileSync(join(process.cwd(), configFile), "utf8");
  const branch = source.match(
    /globalTimeout:\s*process\.env\.CI\s*\?\s*([\w$]+)/,
  );
  if (!branch) {
    throw new Error(
      `Could not read globalTimeout from ${configFile}. If the config changed shape, update this test.`,
    );
  }

  const value = branch[1];
  if (/^\d[\d_]*$/.test(value)) return toMs(value);

  const numericConsts = new Map<string, number>();
  for (const decl of source.matchAll(
    /const\s+([\w$]+)\s*=\s*(\d[\d_]*)\s*;/g,
  )) {
    numericConsts.set(decl[1], toMs(decl[2]));
  }
  if (numericConsts.has(value)) return numericConsts.get(value)!;

  // One hop: find the identifier's declaration and take the first numeric
  // const it refers to — that is the documented default.
  const declaration = source.match(
    new RegExp(`const\\s+${value}\\s*=([\\s\\S]*?);`),
  );
  for (const ref of declaration?.[1].matchAll(/[\w$]+/g) ?? []) {
    const resolved = numericConsts.get(ref[0]);
    if (resolved !== undefined) return resolved;
  }

  throw new Error(
    `globalTimeout in ${configFile} resolves to "${value}", whose value could not be read.`,
  );
}

function specFiles(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) return specFiles(path);
    return entry.name.endsWith(".spec.ts") ? [path] : [];
  });
}

interface Ceiling {
  file: string;
  line: number;
  ms: number;
}

function declaredCeilings(): Ceiling[] {
  return specFiles(SPEC_ROOT).flatMap((file) =>
    readFileSync(file, "utf8")
      .split("\n")
      .flatMap((text, index) => {
        const match = text.match(/test\.setTimeout\(\s*([\d_]+)\s*\)/);
        if (!match) return [];
        return [
          {
            file: file.slice(process.cwd().length + 1),
            line: index + 1,
            ms: Number(match[1].replace(/_/g, "")),
          },
        ];
      }),
  );
}

describe("mock-LLM per-test timeout budget", () => {
  it("reads a global timeout from every config that runs these specs", () => {
    for (const config of CONFIGS) {
      expect(readGlobalTimeoutMs(config), config).toBeGreaterThan(0);
    }
    expect(declaredCeilings().length).toBeGreaterThan(0);
  });

  it("keeps every per-test ceiling within its share of the smallest budget", () => {
    const budget = Math.min(...CONFIGS.map(readGlobalTimeoutMs));
    const limit = budget * MAX_SHARE_OF_GLOBAL_BUDGET;

    const offenders = declaredCeilings()
      .filter((ceiling) => ceiling.ms > limit)
      .map(
        (ceiling) =>
          `${ceiling.file}:${ceiling.line} declares ${ceiling.ms / 1000}s, ` +
          `over the ${limit / 1000}s ceiling ` +
          `(${MAX_SHARE_OF_GLOBAL_BUDGET * 100}% of the ${budget / 1000}s budget)`,
      );

    expect(offenders, offenders.join("\n")).toEqual([]);
  });

  it("keeps each readiness poll inside the ceiling of the test that runs it", () => {
    // A poll that outlasts its own test is unreachable: Playwright ends the
    // test first, replacing a readable "never ready" assertion with an opaque
    // timeout. `test.setTimeout` opens a test body, so attribute each poll to
    // the most recent ceiling above it.
    const offenders: string[] = [];

    for (const file of specFiles(SPEC_ROOT)) {
      let ceiling: { ms: number; line: number } | null = null;

      readFileSync(file, "utf8")
        .split("\n")
        .forEach((text, index) => {
          const declared = text.match(/test\.setTimeout\(\s*([\d_]+)\s*\)/);
          if (declared) {
            ceiling = { ms: toMs(declared[1]), line: index + 1 };
            return;
          }

          const poll = text.match(/pollUrl\([^,]*,\s*([\d_]+)\s*\)/);
          if (!poll || !ceiling) return;

          const ms = toMs(poll[1]);
          if (ms >= ceiling.ms) {
            offenders.push(
              `${file.slice(process.cwd().length + 1)}:${index + 1} polls for ` +
                `${ms / 1000}s inside a test capped at ${ceiling.ms / 1000}s ` +
                `(line ${ceiling.line})`,
            );
          }
        });
    }

    expect(offenders, offenders.join("\n")).toEqual([]);
  });
});
