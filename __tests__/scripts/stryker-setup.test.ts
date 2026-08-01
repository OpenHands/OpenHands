// @vitest-environment node
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const repoRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);

describe("Stryker mutation testing setup", () => {
  it("uses the Vitest runner for first-party TypeScript source", async () => {
    const config = (
      (await import("../../stryker.config.mjs")) as {
        default: {
          mutate: string[];
          testRunner: string;
          vitest: { configFile: string; related: boolean };
        };
      }
    ).default;

    expect(config).toMatchObject({
      testRunner: "vitest",
      vitest: {
        configFile: "vite.config.ts",
        related: true,
      },
    });
    expect(config.mutate).toContain("src/**/*.{ts,tsx}");
    expect(config.mutate).toContain("!src/**/*.{test,spec}.{ts,tsx}");
    expect(config.mutate).toContain("!src/**/*.d.ts");
    expect(config.mutate).toContain("!src/{fixtures,mocks,dev}/**");
  });

  it("provides full, incremental, and branch-diff commands", () => {
    const packageJson = JSON.parse(
      readFileSync(path.join(repoRoot, "package.json"), "utf8"),
    ) as {
      devDependencies: Record<string, string>;
      scripts: Record<string, string>;
    };

    expect(packageJson.scripts).toMatchObject({
      "test:mutation": "npm run make-i18n && stryker run",
      "test:mutation:diff":
        "npm run make-i18n && node scripts/stryker-diff.mjs",
      "test:mutation:incremental":
        "npm run make-i18n && stryker run --incremental",
    });
    expect(packageJson.devDependencies).toMatchObject({
      "@stryker-mutator/core": "9.6.1",
      "@stryker-mutator/vitest-runner": "9.6.1",
    });
  });

  it("ignores local mutation sandboxes and reports", () => {
    const gitignore = readFileSync(path.join(repoRoot, ".gitignore"), "utf8");

    expect(gitignore).toContain(".stryker-tmp/");
    expect(gitignore).toContain("reports/");
  });
});
