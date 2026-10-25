// @vitest-environment node
import viteConfig from "../vite.config";
import { build } from "vite";
import { execFileSync } from "node:child_process";
import { mkdtemp, writeFile, symlink, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { Script } from "node:vm";
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
    // Scoped to the Canvas root so it never restyles the host page.
    expect(css).toContain("[data-agent-server-ui] .flex{display:flex}");
    expect(css).not.toMatch(/(^|[{},])\.flex\{/);
    expect(css).toContain("[data-agent-server-ui] .xterm");
    expect(css).toContain("fonts.googleapis.com");

    // Extension dialogs keep their viewport limits in the shipped library.
    const dialogViewportRule =
      css.match(
        /\[data-agent-server-ui\]\s+\.extension-dialog-viewport\s*\{([^}]*)\}/,
      )?.[1] ?? "";
    const dialogViewportDeclarations = Object.fromEntries(
      dialogViewportRule
        .split(";")
        .filter((declaration) => declaration.trim())
        .map((declaration) =>
          declaration.split(":").map((part) => part.trim()),
        ),
    );
    expect(dialogViewportDeclarations).toMatchObject({
      "max-width": "90vw",
      "max-height": "85vh",
    });
    expect(css).not.toMatch(/(^|[{},])\.extension-dialog-viewport\{/);

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
