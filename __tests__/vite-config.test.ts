// @vitest-environment node
import viteConfig from "../vite.config";
import { build } from "vite";
import { execFileSync } from "node:child_process";
import { mkdtemp, writeFile, symlink, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { Script } from "node:vm";
import { parse, type AnyNode, type AtRule, type Rule } from "postcss";
import { afterEach, describe, expect, it } from "vitest";

afterEach(() => {
  delete process.env.BUILD_LIB;
});

describe("vite optimizeDeps", () => {
  it("prebundles core client entry dependencies", async () => {
    const config = await viteConfig({ mode: "development", command: "serve" });
    const optimizedDeps = config.optimizeDeps?.include ?? [];

    expect(optimizedDeps).toEqual(
      expect.arrayContaining([
        "react",
        "react/jsx-runtime",
        "react-dom/client",
        "react-router/dom",
      ]),
    );
  });
});

describe("vitest environment projects", () => {
  const getProjects = async () => {
    const config = await viteConfig({ mode: "test", command: "serve" });

    return (config.test?.projects ?? []) as Array<{
      test?: {
        name?: string;
        environment?: string;
        setupFiles?: string[];
        include?: string[];
        exclude?: string[];
      };
    }>;
  };

  const findProject = (
    projects: Awaited<ReturnType<typeof getProjects>>,
    name: string,
  ) => projects.find((project) => project.test?.name === name)?.test;

  it("runs the DOM-free pilot suites in the Node environment without the jsdom setup", async () => {
    const nodeProject = findProject(await getProjects(), "node");

    expect(nodeProject?.environment).toBe("node");
    // The pilot suites are pure functions: no DOM, no Web Storage, no MSW.
    expect(nodeProject?.setupFiles).toEqual([]);
    expect(nodeProject?.include).toEqual([
      "__tests__/utils/file-language.test.ts",
      "__tests__/utils/format-model-name.test.ts",
      "__tests__/utils/parse-terminal-output.test.ts",
    ]);
  });

  it("keeps every other suite in jsdom with the shared setup", async () => {
    const jsdomProject = findProject(await getProjects(), "jsdom");

    expect(jsdomProject?.environment).toBe("jsdom");
    expect(jsdomProject?.setupFiles).toEqual(["vitest.setup.ts"]);
  });

  it("excludes the pilot suites from jsdom so they run exactly once", async () => {
    const projects = await getProjects();
    const nodeInclude = findProject(projects, "node")?.include ?? [];
    const jsdomExclude = findProject(projects, "jsdom")?.exclude ?? [];

    expect(nodeInclude.length).toBeGreaterThan(0);
    expect(jsdomExclude).toEqual(expect.arrayContaining(nodeInclude));
  });
});

describe("vite path resolution", () => {
  it("uses Vite's native tsconfig paths support", async () => {
    const config = await viteConfig({ mode: "development", command: "serve" });

    expect(config.resolve?.tsconfigPaths).toBe(true);
    expect(config.plugins).not.toEqual(
      expect.arrayContaining([
        expect.objectContaining({ name: "vite-tsconfig-paths" }),
      ]),
    );
  });
});

describe("vite app build", () => {
  it("configures Rolldown code splitting for large vendor chunks", async () => {
    const config = await viteConfig({ mode: "production", command: "build" });
    const appBuild = config as {
      build?: {
        rolldownOptions?: {
          output?: {
            codeSplitting?: {
              groups?: Array<{
                name?: string;
                maxSize?: number;
                entriesAware?: boolean;
              }>;
            };
          };
        };
      };
    };

    expect(
      appBuild.build?.rolldownOptions?.output?.codeSplitting?.groups,
    ).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          name: "vendor",
          maxSize: 450 * 1024,
          entriesAware: true,
        }),
      ]),
    );
  });
});

describe("vite library build", () => {
  it("configures a dual-format preserved-module library build", async () => {
    process.env.BUILD_LIB = "true";

    const config = await viteConfig({ mode: "production", command: "build" });

    expect((config as { copyPublicDir?: boolean }).copyPublicDir).toBe(false);
    expect(config.build?.lib).toMatchObject({
      formats: ["es"],
    });
    expect(config.build?.rollupOptions?.output).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          format: "es",
          preserveModules: true,
          preserveModulesRoot: "src",
        }),
        expect.objectContaining({
          format: "cjs",
          preserveModules: true,
          preserveModulesRoot: "src",
          exports: "named",
        }),
      ]),
    );
  });

  it("emits the scoped Canvas stylesheet and a parseable CJS entry", async () => {
    process.env.BUILD_LIB = "true";
    const config = await viteConfig({ mode: "production", command: "build" });
    const outputs = config.build?.rollupOptions?.output;
    const cjsOutput = (Array.isArray(outputs) ? outputs : [outputs]).find(
      (output) => output?.format === "cjs",
    );
    if (!cjsOutput) throw new Error("Expected a CJS library output");
    const built = await build({
      ...config,
      configFile: false,
      logLevel: "error",
      build: {
        ...config.build,
        sourcemap: false,
        write: false,
        rollupOptions: { ...config.build?.rollupOptions, output: [cjsOutput] },
      },
    });
    const [result] = Array.isArray(built) ? built : [built];
    if (!("output" in result))
      throw new Error("Expected a completed library build");
    const file = (fileName: string) => {
      const item = result.output.find((entry) => entry.fileName === fileName);
      if (!item) throw new Error(`Missing ${fileName}`);
      return String(item.type === "asset" ? item.source : item.code);
    };

    const css = file("styles.css");
    const stylesheet = parse(css);
    const declarations = (rule: Rule) => {
      const values: Record<string, string> = {};
      rule.walkDecls(({ prop, value }) => {
        values[prop] = value;
      });
      return values;
    };
    const recipe = (name: string, pseudo = "") => {
      const classSelector = `.${name.replace(/([^\w-])/g, "\\$1")}${pseudo}`;
      const matches: Rule[] = [];
      stylesheet.walkRules((rule) => {
        if (
          rule.selectors.some(
            (selector) =>
              selector === classSelector ||
              selector.endsWith(` ${classSelector}`),
          )
        ) {
          matches.push(rule);
        }
      });
      expect(matches, `Shipped scoped recipe: ${name}`).toHaveLength(1);
      const rule = matches[0];
      expect(rule.selectors).toContain(
        `[data-agent-server-ui] ${classSelector}`,
      );
      for (const selector of rule.selectors) {
        expect(selector).toMatch(/^\[data-agent-server-ui\]\s/);
      }
      let inUtilitiesLayer = false;
      for (
        let parent: AnyNode | undefined = rule.parent;
        parent;
        parent = parent.parent
      ) {
        if (
          parent.type === "atrule" &&
          (parent as AtRule).name === "layer" &&
          (parent as AtRule).params === "utilities"
        ) {
          inUtilitiesLayer = true;
        }
      }
      expect(inUtilitiesLayer, `Utilities-layer recipe: ${name}`).toBe(true);
      return rule;
    };
    // Scoped to the Canvas root so it never restyles the host page.
    expect(css).toContain("[data-agent-server-ui] .flex{display:flex}");
    expect(css).not.toMatch(/(^|[{},])\.flex\{/);
    expect(css).toContain("[data-agent-server-ui] .xterm");
    expect(css).toContain("fonts.googleapis.com");

    // Extension dialogs keep their viewport limits in the shipped library.
    expect(declarations(recipe("extension-dialog-viewport"))).toMatchObject({
      "max-width": "90vw",
      "max-height": "85vh",
    });
    expect(css).not.toMatch(/(^|[{},])\.extension-dialog-viewport\{/);

    // Require the setup rail and editor geometry in the extracted stylesheet;
    // every recipe must remain a scoped utility even inside media conditions.
    expect(declarations(recipe("automation-setup-content"))).toMatchObject({
      "max-width": "800px",
    });
    expect(
      declarations(recipe("automation-setup-scroll-gutter")),
    ).toMatchObject({
      "scrollbar-gutter": "stable",
    });
    expect(declarations(recipe("automation-setup-card"))).toMatchObject({
      "border-radius": "15px",
    });
    expect(declarations(recipe("automation-setup-joined-card"))).toMatchObject({
      "border-radius": "15px",
      "margin-bottom": "-15px",
    });
    const tray = declarations(recipe("automation-setup-joined-tray"));
    expect(tray).toMatchObject({
      "border-bottom-right-radius": "15px",
      "border-bottom-left-radius": "15px",
    });
    expect(tray["padding-top"].replace(/\s/g, "")).toMatch(
      /^calc\(15px\+0?\.5rem\)$/,
    );
    const dock = declarations(recipe("automation-setup-composer-dock"));
    expect(dock["border-radius"]).toBe("15px");
    expect(dock["--tw-shadow"]).toContain("0 12px 40px");
    expect(dock["box-shadow"]).toMatch(
      /var\(--tw-ring-shadow\).*var\(--tw-shadow\)/,
    );
    expect(declarations(recipe("automation-setup-prompt-input"))).toMatchObject(
      {
        "min-height": "120px",
      },
    );
    expect(declarations(recipe("automation-setup-code-scroll"))).toMatchObject({
      "min-height": "256px",
    });

    // Setup pickers preserve their fixed and viewport-relative limits.
    expect(
      declarations(recipe("automation-setup-model-trigger")),
    ).toMatchObject({
      "max-width": "200px",
    });
    expect(
      declarations(recipe("automation-setup-model-popover")),
    ).toMatchObject({
      "min-width": "200px",
      "max-width": "320px",
      "max-height": "60vh",
    });
    expect(
      declarations(recipe("automation-setup-repository-dialog")),
    ).toMatchObject({
      width: "520px",
      "max-width": "90vw",
      "max-height": "85vh",
    });
    const repositoryMenu = declarations(
      recipe("automation-setup-repository-menu"),
    );
    expect(repositoryMenu.width.replace(/\s/g, "")).toBe(
      "min(28rem,100vw-3rem)",
    );

    // Fields consume row-owned rem widths after merging shared form classes.
    const scheduleRow = recipe("automation-setup-schedule-row");
    const scheduleWidths = {
      "--automation-setup-cron-width": "14rem",
      "--automation-setup-datetime-width": "15rem",
      "--automation-setup-time-width": "9.5rem",
      "--automation-setup-timezone-width": "16rem",
    };
    expect(declarations(scheduleRow)).toMatchObject(scheduleWidths);
    for (const property of Object.keys(scheduleWidths)) {
      expect(declarations(recipe(`w-(${property})`))).toMatchObject({
        width: `var(${property})`,
      });
      const owners: Rule[] = [];
      stylesheet.walkDecls(property, (declaration) => {
        if (declaration.parent?.type === "rule") {
          owners.push(declaration.parent as Rule);
        }
      });
      expect(owners).toHaveLength(1);
      expect(owners[0]).toBe(scheduleRow);
    }
    const grid = recipe("md:automation-setup-details-grid");
    expect(declarations(grid)["grid-template-columns"]).toBe("2fr 1fr");
    expect(grid.parent).toMatchObject({
      type: "atrule",
      name: "media",
      params: expect.stringMatching(/^\(width\s*>=\s*48rem\)$/),
    });

    // Connection chooser disclosures retain exact fractional tracks and their
    // motion override after extraction and library selector scoping.
    const disclosure = recipe("backend-chooser-disclosure");
    expect(declarations(disclosure)).toMatchObject({
      "transition-property": "grid-template-rows",
      "transition-timing-function":
        "var(--tw-ease,var(--default-transition-timing-function))",
      "transition-duration":
        "var(--tw-duration,var(--default-transition-duration))",
    });
    expect(
      declarations(recipe("backend-chooser-disclosure-open")),
    ).toMatchObject({ "grid-template-rows": "1fr" });
    expect(
      declarations(recipe("backend-chooser-disclosure-closed")),
    ).toMatchObject({ "grid-template-rows": "0fr" });
    expect(declarations(recipe("duration-200"))).toMatchObject({
      "transition-duration": ".2s",
    });
    const easeOut = declarations(recipe("ease-out"));
    expect(easeOut["transition-timing-function"]).toBe("var(--ease-out)");
    const easingValues: string[] = [];
    stylesheet.walkDecls("--ease-out", ({ value }) => {
      easingValues.push(value.replace(/\s/g, ""));
    });
    expect(easingValues).toContain("cubic-bezier(0,0,.2,1)");
    const reducedMotion = recipe("motion-reduce:transition-none");
    expect(declarations(reducedMotion)["transition-property"]).toBe("none");
    expect(reducedMotion.parent).toMatchObject({
      type: "atrule",
      name: "media",
      params: "(prefers-reduced-motion:reduce)",
    });
    expect(reducedMotion.source?.start?.offset).toBeGreaterThan(
      disclosure.source?.start?.offset ?? Infinity,
    );

    // Preserve relative typography/geometry and distinct Add/Edit viewport
    // semantics, including the Cloud panel's grayscale pseudo-element halo.
    expect(declarations(recipe("backend-chooser-cloud-panel"))).toMatchObject({
      "min-height": "13.5rem",
    });
    const halo = declarations(
      recipe("before:backend-chooser-cloud-halo", ":before"),
    );
    expect(halo["background-image"]).toBe(
      "radial-gradient(75% 75%,#ffffff1a 0%,#fff0 70%)",
    );
    expect(halo.content).toBe("var(--tw-content)");
    expect(
      declarations(recipe("backend-chooser-learn-more-icon")),
    ).toMatchObject({ "vertical-align": "-.125em" });
    expect(declarations(recipe("backend-chooser-add-dialog"))).toMatchObject({
      "max-height": "92vh",
    });
    expect(declarations(recipe("backend-chooser-edit-dialog"))).toMatchObject({
      "max-height": "92dvh",
    });

    // `package.json` routes `require` here; stripped CSS imports must not
    // leave it unparseable.
    expect(() => new Script(file("index.cjs"))).not.toThrow();
  }, 180000);
});

it("emits browser-compatible ESM when a bundled CommonJS dependency requires a React peer", async () => {
  process.env.BUILD_LIB = "true";
  const directory = await mkdtemp(join(tmpdir(), "canvas-esm-peer-"));
  try {
    await writeFile(
      join(directory, "package.json"),
      JSON.stringify({ type: "module" }),
    );
    await symlink(
      resolve("node_modules"),
      join(directory, "node_modules"),
      "dir",
    );
    await writeFile(
      join(directory, "entry.js"),
      'export { useSyncExternalStore } from "use-sync-external-store/shim"; export { motion } from "framer-motion";',
    );
    const config = await viteConfig({ mode: "production", command: "build" });
    const built = await build({
      ...config,
      configFile: false,
      logLevel: "error",
      root: directory,
      build: {
        ...config.build,
        sourcemap: false,
        lib: { entry: join(directory, "entry.js"), formats: ["es"] },
        outDir: join(directory, "dist"),
      },
    });
    const outputs = Array.isArray(built) ? built : [built];
    const esm = outputs[0];
    if (!("output" in esm))
      throw new Error("Expected a completed library build");
    const entry = esm.output.find(
      (chunk) => chunk.type === "chunk" && chunk.isEntry,
    );
    if (!entry) throw new Error("Library entry was not emitted");
    await writeFile(
      join(directory, "consumer.mjs"),
      `import React from "react"; import { motion as hostMotion } from "framer-motion"; import { useSyncExternalStore, motion } from ${JSON.stringify("./dist/" + entry.fileName)}; if (useSyncExternalStore !== React.useSyncExternalStore) throw new Error("React peer hook was not preserved"); if (motion !== hostMotion) throw new Error("Motion dependency was not preserved");`,
    );

    // A native ESM consumer has no CommonJS require shim, like a browser host.
    execFileSync(process.execPath, [join(directory, "consumer.mjs")], {
      stdio: "pipe",
    });
    const cjs = outputs[1];
    if (!("output" in cjs)) throw new Error("Expected CommonJS library output");
    const commonEntry = cjs.output.find(
      (chunk) => chunk.type === "chunk" && chunk.isEntry,
    );
    if (!commonEntry) throw new Error("CommonJS entry was not emitted");
    await writeFile(
      join(directory, "consumer.cjs"),
      `const React = require("react"); const { motion: hostMotion } = require("framer-motion"); const { useSyncExternalStore, motion } = require(${JSON.stringify("./dist/" + commonEntry.fileName)}); if (useSyncExternalStore !== React.useSyncExternalStore) throw new Error("CommonJS React peer hook was not preserved"); if (motion !== hostMotion) throw new Error("CommonJS Motion dependency was not preserved");`,
    );
    execFileSync(process.execPath, [join(directory, "consumer.cjs")], {
      stdio: "pipe",
    });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
}, 30000);
