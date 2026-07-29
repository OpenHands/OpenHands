// @vitest-environment node
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const packageJson = JSON.parse(
  readFileSync(resolve(__dirname, "../package.json"), "utf8"),
) as {
  name: string;
  main: string;
  module: string;
  types: string;
  exports: Record<string, unknown>;
  scripts: Record<string, string>;
  dependencies?: Record<string, string>;
  devDependencies?: Record<string, string>;
};

const EXACT_SEMVER_PATTERN =
  /^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$/;

describe("package library metadata", () => {
  const ALLOWED_STACK_PIN_DEPS = new Set([
    "@openhands/extensions",
    "@openhands/typescript-client",
  ]);

  it("publishes the agent-canvas package entrypoints", () => {
    expect(packageJson.name).toBe("@openhands/agent-canvas");
    expect(packageJson.main).toBe("./dist/index.cjs");
    expect(packageJson.module).toBe("./dist/index.js");
    expect(packageJson.types).toBe("./dist/index.d.ts");
    expect(packageJson.exports).toMatchObject({
      ".": {
        types: "./dist/index.d.ts",
        import: "./dist/index.js",
        require: "./dist/index.cjs",
      },
      "./conversation": {
        types: "./dist/components/conversation/index.d.ts",
        import: "./dist/components/conversation/index.js",
        require: "./dist/components/conversation/index.cjs",
      },
      "./settings": {
        types: "./dist/components/settings/index.d.ts",
        import: "./dist/components/settings/index.js",
        require: "./dist/components/settings/index.cjs",
      },
      "./terminal": {
        types: "./dist/components/terminal/index.d.ts",
        import: "./dist/components/terminal/index.js",
        require: "./dist/components/terminal/index.cjs",
      },
      "./i18n": {
        types: "./dist/i18n/index.d.ts",
        import: "./dist/i18n/index.js",
        require: "./dist/i18n/index.cjs",
      },
    });
  });

  // Git dependencies break `npm install -g` because npm clones the repo and
  // runs the prepare script without devDependencies. All packages should be
  // referenced from a registry. @openhands/extensions is allowed until it is
  // published to npm; @openhands/typescript-client is temporarily allowed while
  // this stacked PR waits for the subscription client branch to merge/release.
  // TODO(#917): remove @openhands/typescript-client exemption once
  // OpenHands/typescript-client#178 merges and publishes to npm.
  it("does not use git dependencies except approved stack pins", () => {
    const GIT_DEP_PATTERN =
      /^(git[+:]|github:|bitbucket:|gitlab:|[a-zA-Z0-9_-]+\/)/;
    const allDeps = {
      ...packageJson.dependencies,
      ...packageJson.devDependencies,
    };

    const violations = Object.entries(allDeps)
      .filter(
        ([name, version]) =>
          GIT_DEP_PATTERN.test(version) && !ALLOWED_STACK_PIN_DEPS.has(name),
      )
      .map(([name, version]) => `${name}: ${version}`);

    expect(violations).toEqual([]);
  });

  it("pins direct dependency versions exactly", () => {
    const allDepsBySection = {
      dependencies: packageJson.dependencies,
      devDependencies: packageJson.devDependencies,
    };

    const violations = Object.entries(allDepsBySection).flatMap(
      ([section, dependencies]) =>
        Object.entries(dependencies ?? {})
          .filter(
            ([name, version]) =>
              !EXACT_SEMVER_PATTERN.test(version) &&
              !ALLOWED_STACK_PIN_DEPS.has(name),
          )
          .map(([name, version]) => `${section}.${name}: ${version}`),
    );

    expect(violations).toEqual([]);
  });

  it("ships runtime logger dependencies for the published CLI", () => {
    expect(packageJson.dependencies).toMatchObject({
      winston: "3.19.0",
      "winston-daily-rotate-file": "5.0.0",
    });
    expect(packageJson.devDependencies?.winston).toBeUndefined();
    expect(
      packageJson.devDependencies?.["winston-daily-rotate-file"],
    ).toBeUndefined();
  });

  it("uses local dev commands without Docker", () => {
    expect(packageJson.scripts.dev).toBe(
      "node --env-file-if-exists=.env scripts/dev-with-automation.mjs",
    );
    expect(packageJson.scripts["dev:static"]).toBe(
      "node --env-file-if-exists=.env scripts/dev-static.mjs",
    );
    expect(packageJson.scripts["dev:minimal"]).toBe(
      "node --env-file-if-exists=.env scripts/dev-safe.mjs",
    );
    expect(packageJson.scripts["dev:docker"]).toBeUndefined();
    expect(packageJson.scripts["dev:docker:dynamic"]).toBeUndefined();
    expect(packageJson.scripts["dev:dangerously-dockerless"]).toBeUndefined();
  });
});
