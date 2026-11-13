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
        "react-aria",
        "react-aria-components",
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
    // The library inlines the local font; its license must ship alongside it.
    const fontLicense = file("assets/newsreader-OFL.txt");
    expect(fontLicense).toContain(
      "Copyright 2020 The Newsreader Project Authors",
    );
    expect(fontLicense).toContain("SIL OPEN FONT LICENSE Version 1.1");
    const stylesheet = parse(css);
    const declarations = (rule: Rule) => {
      const values: Record<string, string> = {};
      rule.walkDecls(({ prop, value }) => {
        values[prop] = value;
      });
      return values;
    };
    const recipes = (name: string, pseudo = "") => {
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
      expect(matches.length, `Shipped scoped recipe: ${name}`).toBeGreaterThan(
        0,
      );
      for (const rule of matches) {
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
      }
      return matches;
    };
    const recipe = (name: string, pseudo = "") => {
      const matches = recipes(name, pseudo);
      expect(matches, `Single scoped recipe: ${name}`).toHaveLength(1);
      return matches[0];
    };
    // These aliases resolve the exact public Canvas variables, including host overrides.
    expect(declarations(recipe("bg-canvas-background"))).toEqual({
      "background-color": "var(--oh-background)",
    });
    expect(declarations(recipe("text-on-accent"))).toEqual({
      color: "var(--oh-accent-foreground)",
    });
    expect(declarations(recipe("file-strip-scrollbar-hidden"))).toMatchObject({
      "scrollbar-width": "none",
      "-ms-overflow-style": "none",
    });
    expect(
      declarations(
        recipe("file-strip-scrollbar-hidden", "::-webkit-scrollbar"),
      ),
    ).toMatchObject({ display: "none" });
    expect(declarations(recipe("file-discovery-viewport-height"))).toEqual({
      "max-height": "90vh",
    });

    // A named gradient stop retains the existing ancestor override/fallback.
    const scrollFade = recipe("from-scroll-fade");
    expect(declarations(scrollFade)["--tw-gradient-from"]).toBe(
      "var(--oh-scroll-fade-from,var(--oh-color-base))",
    );
    expect(declarations(recipe("scroll-fade-surface"))).toEqual({
      "--oh-scroll-fade-from": "var(--oh-surface)",
    });
    expect(declarations(recipe("scroll-fade-tertiary"))).toEqual({
      "--oh-scroll-fade-from": "var(--oh-color-tertiary)",
    });
    // Keep code padding relative to its own font size in embedded roots.
    expect(declarations(recipe("px-code-inline-x"))).toEqual({
      "padding-inline": ".4em",
    });
    expect(declarations(recipe("py-code-inline-y"))).toEqual({
      "padding-block": ".2em",
    });
    expect(declarations(recipe("p-code-block"))).toEqual({ padding: "1em" });
    // The drop illustration uses fixed SVG pixel coordinates even in embeds.
    expect(declarations(recipe("file-drop-image"))).toEqual({
      top: "-40px",
      left: "-10px",
      rotate: "-27deg",
    });
    expect(declarations(recipe("file-drop-arrow"))).toEqual({
      top: "-16px",
      left: "-20px",
    });
    // Typography recipes retain signed values and their original units.
    for (const [name, value] of [
      ["tracking-metadata", ".11px"],
      ["tracking-label", ".01em"],
      ["tracking-title", "-.01em"],
      ["tracking-heading", "-.02em"],
      ["tracking-auth-code", ".08em"],
      ["tracking-device-code", ".3em"],
    ]) {
      expect(declarations(recipe(name))).toEqual({
        "--tw-tracking": value,
        "letter-spacing": value,
      });
    }
    // Message wrapping must retain the legacy word-break behavior; the standard
    // overflow utility controls the independent min-content wrapping contract.
    expect(declarations(recipe("message-word-break"))).toEqual({
      "word-break": "break-word",
    });
    expect(declarations(recipe("wrap-anywhere"))).toEqual({
      "overflow-wrap": "anywhere",
    });
    // Scoped to the Canvas root so it never restyles the host page.
    expect(css).toContain("[data-agent-server-ui] .flex{display:flex}");
    expect(css).not.toMatch(/(^|[{},])\.flex\{/);
    expect(css).toContain("[data-agent-server-ui] .xterm");
    expect(css).toContain("fonts.googleapis.com");

    // Inset recipes retain direction, absolute pixels and public color hooks.
    expect(
      recipes("shadow-badge-inset")
        .map((rule) => declarations(rule)["--tw-shadow"]?.replace(/,\s+/g, ","))
        .filter(Boolean)
        .at(-1),
    ).toBe(
      "inset 0 1px 0 var(--tw-shadow-color,color-mix(in srgb,var(--oh-contrast) 18%,transparent))",
    );
    for (const [name, offset] of [
      ["shadow-reorder-before", "2px"],
      ["shadow-reorder-after", "-2px"],
    ]) {
      expect(declarations(recipe(name))["--tw-shadow"]).toBe(
        `inset 0 ${offset} 0 0 var(--tw-shadow-color,var(--oh-focus))`,
      );
    }
    // Extension dialogs keep their viewport limits in the shipped library.
    expect(declarations(recipe("extension-dialog-viewport"))).toMatchObject({
      "max-width": "90vw",
      "max-height": "85vh",
    });
    expect(css).not.toMatch(/(^|[{},])\.extension-dialog-viewport\{/);

    expect(declarations(recipe("w-commit-sha")).width).toBe("7ch");
    expect(declarations(recipe("h-(--editor-height)")).height).toBe(
      "var(--editor-height)",
    );

    // Require the setup rail and editor geometry in the extracted stylesheet;
    // every recipe must remain a scoped utility even inside media conditions.
    expect(declarations(recipe("automation-setup-content"))).toMatchObject({
      "max-width": "56rem",
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
      "--automation-setup-timezone-width": "12.5rem",
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

    // The sidebar shortcut ships to hosts; palette recipes also stay scoped
    // for the standalone body portal without changing its mount boundary.
    expect(declarations(recipe("z-70"))["z-index"]).toBe("70");
    expect(declarations(recipe("command-menu-overlay"))["padding-top"]).toBe(
      "10vh",
    );
    const commandPanel = declarations(recipe("command-menu-panel"));
    expect(commandPanel["max-height"].replace(/\s/g, "")).toBe(
      "min(720px,78vh)",
    );
    expect(commandPanel["--tw-shadow"]).toContain("0 24px 90px");
    expect(commandPanel["--tw-shadow"]).toContain("inset");
    // Vite may split color fallbacks into @supports rules. Check each
    // emitted selector's scope and collect all supported declarations.
    const commandSelection = Object.assign(
      {},
      ...recipes("command-menu-active-option").map(declarations),
    );
    expect(commandSelection["--tw-shadow"]).toContain("var(--oh-contrast)");
    expect(commandSelection["--tw-shadow"]).toContain("8%");
    for (const shadow of [commandPanel, commandSelection]) {
      for (const variable of [
        "--tw-inset-shadow",
        "--tw-inset-ring-shadow",
        "--tw-ring-offset-shadow",
        "--tw-ring-shadow",
        "--tw-shadow",
      ]) {
        expect(shadow["box-shadow"]).toContain(`var(${variable})`);
      }
    }
    const commandBackdrop = declarations(recipe("command-menu-backdrop"));
    expect(commandBackdrop["--tw-backdrop-blur"]).toBe("blur(2px)");
    expect(commandBackdrop["backdrop-filter"]).toContain(
      "var(--tw-backdrop-blur,)",
    );
    expect(commandBackdrop["backdrop-filter"]).toContain(
      "var(--tw-backdrop-brightness,)",
    );
    expect(commandBackdrop["-webkit-backdrop-filter"]).toBe(
      commandBackdrop["backdrop-filter"],
    );
    for (const [name, fontSize, tracking] of [
      ["command-menu-label", "11px", 0.18],
      ["command-menu-result-hint", "10px", 0.14],
      ["command-menu-trigger-hint", "10px", 0.12],
      ["command-menu-footer", "11px", undefined],
    ] as const) {
      const typography = declarations(recipe(name));
      expect(typography["font-size"]).toBe(fontSize);
      expect(typography["line-height"]).toBeUndefined();
      if (tracking === undefined) {
        expect(typography["letter-spacing"]).toBeUndefined();
      } else {
        expect(typography["letter-spacing"]).toMatch(/em$/);
        expect(Number.parseFloat(typography["letter-spacing"])).toBe(tracking);
        expect(typography["--tw-tracking"]).toBe(typography["letter-spacing"]);
      }
    }

    const contextTrackMix =
      "color-mix(in srgb,var(--oh-context-window-foreground) var(--oh-context-window-track-weight),transparent)";
    for (const [className, property] of [
      ["stroke-context-window-track", "stroke"],
      ["border-context-window-track", "border-color"],
    ] as const) {
      const track = Object.assign({}, ...recipes(className).map(declarations));
      expect(track[property]?.replace(/,\s*/g, ",")).toBe(contextTrackMix);
    }
    expect(declarations(recipe("w-(--context-fill-width)")).width).toBe(
      "var(--context-fill-width)",
    );

    // Connection chooser disclosures retain exact fractional tracks and their
    // motion override after extraction and library selector scoping.
    const transitionProperties = {
      "transition-control-shell":
        "background-color,border-color,box-shadow,opacity",
      "!transition-control-shell":
        "background-color,border-color,box-shadow,opacity",
      "transition-control-transform": "transform",
      "transition-context-ring": "stroke-dashoffset,stroke",
      "transition-panel-width": "width",
      "transition-sidebar-width": "width,min-width",
    };
    for (const [name, properties] of Object.entries(transitionProperties)) {
      const motion = recipe(name);
      expect(declarations(motion)).toEqual({
        "transition-property": properties,
        "transition-timing-function":
          "var(--tw-ease,var(--default-transition-timing-function))",
        "transition-duration":
          "var(--tw-duration,var(--default-transition-duration))",
      });
      motion.walkDecls((declaration) => {
        expect(Boolean(declaration.important)).toBe(name.startsWith("!"));
      });
    }
    expect(declarations(recipe("animate-chat-status-pulse"))).toEqual({
      animation: "1.2s ease-in-out infinite pulse",
    });
    const pulseFrames: Record<string, Record<string, string>> = {};
    stylesheet.walkAtRules("keyframes", (frames) => {
      if (frames.params === "pulse")
        frames.walkRules((frame) => {
          pulseFrames[frame.selector] = declarations(frame);
        });
    });
    expect(pulseFrames).toEqual({ "50%": { opacity: ".5" } });

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

    expect(
      declarations(recipe("dark:border-git-control", ":is(.dark *)")),
    ).toEqual({ "border-color": "#474a5480" });

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

    // Workspace selection keeps its mixed rem/px columns and viewport bounds.
    const folderColumns = recipe("workspace-folder-columns");
    expect(declarations(folderColumns)["grid-template-columns"]).toBe(
      "minmax(0,1fr) 4rem",
    );
    const desktopColumns = recipe("md:workspace-folder-columns-desktop");
    expect(declarations(desktopColumns)["grid-template-columns"]).toBe(
      "1fr 120px",
    );
    expect(desktopColumns.parent).toMatchObject({
      type: "atrule",
      name: "media",
      params: "(width>=48rem)",
    });
    expect(declarations(recipe("workspace-section-label"))).toMatchObject({
      "font-size": "11px",
    });
    expect(declarations(recipe("workspace-manager-dialog"))).toMatchObject({
      "max-height": "70vh",
    });
    expect(declarations(recipe("z-9999"))).toMatchObject({
      "z-index": "9999",
    });
    expect(declarations(recipe("home-dropdown-sticky-list"))).toMatchObject({
      "max-height": "12rem",
    });

    // Floating surfaces retain their stacking and compiled SVG stroke width.
    expect(declarations(recipe("z-200"))).toMatchObject({ "z-index": "200" });
    expect(declarations(recipe("context-menu-icon-stroke"))).toMatchObject({
      "stroke-width": "1.75px",
    });
    expect(declarations(recipe("catalog-remove-icon-stroke"))).toMatchObject({
      "stroke-width": "2.5px",
    });

    // Preserve exact menu units and text metrics in the shipped scoped CSS.
    const menuGeometryContracts = {
      "tools-menu-offset": {
        left: "-16px",
      },
      "tools-submenu-offset": {
        top: "-4px",
      },
      "tools-git-submenu-offset": {
        top: "-6px",
      },
      "profile-menu-height": {
        "max-height": "60vh",
      },
      "menu-caption-text": {
        "font-size": "11px",
      },
      "z-60": {
        "z-index": "60",
      },
      "new-conversation-menu-height": {
        "max-height": "40vh",
      },
      "menu-detail-text": {
        "font-size": "10px",
      },
      "conversation-card-menu-width": {
        "max-width": "min(280px,100vw - 16px)",
      },
      "-z-1": {
        "z-index": "calc(1 * -1)",
      },
      "conversation-tag-popover-width": {
        "min-width": "10rem",
        "max-width": "16rem",
      },
      "conversation-tag-popover-columns": {
        "grid-template-columns": "minmax(0,auto) minmax(0,1fr)",
      },
      "conversation-meta-text": {
        "font-size": "10px",
      },
      "skill-pill-popover-width": {
        "max-width": "20rem",
      },
      "conversation-git-actions-width": {
        "min-width": "8rem",
      },
      "automation-actions-menu-width": {
        "min-width": "10rem",
      },
      "dropdown-menu-width": {
        "max-width": "min(26rem,100vw - 2rem)",
      },
      "dropdown-menu-viewport-height": {
        "max-height": "min(60vh,100dvh - 5rem)",
      },
    };
    for (const [name, expected] of Object.entries(menuGeometryContracts)) {
      expect(declarations(recipe(name))).toMatchObject(expected);
    }

    const dialogContracts = {
      "dialog-viewport-width": {
        "max-width": "90vw",
      },
      "onboarding-viewport-height": {
        "max-height": "90vh",
      },
      "dialog-content-height": {
        "max-height": "80vh",
      },
      "setup-dialog-height": {
        "max-height": "85vh",
      },
      "setup-dialog-width": {
        width: "92vw",
      },
      "automation-import-width": {
        width: "min(36rem,100vw - 2rem)",
      },
      "dialog-scroll-height": {
        "max-height": "60vh",
      },
      "automation-log-min-height": {
        "min-height": "12rem",
      },
      "conversation-dialog-reading-height": {
        height: "60vh",
      },
      "conversation-options-bleed-width": {
        width: "calc(100% + 3rem)",
      },
    };
    for (const [name, expected] of Object.entries(dialogContracts)) {
      expect(declarations(recipe(name))).toMatchObject(expected);
    }

    // Compact type tokens must not introduce a line-height default.
    expect(declarations(recipe("text-2xs"))).toEqual({ "font-size": "11px" });
    expect(declarations(recipe("text-3xs"))).toEqual({ "font-size": "10px" });

    for (const [name, size] of Object.entries({
      "text-4xs": "9px",
      "text-xs-plus": "13px",
      "text-sm-plus": "15px",
      "text-title": "32px",
    })) {
      expect(declarations(recipe(name))).toEqual({ "font-size": size });
    }

    expect(declarations(recipe("metadata-grid-columns"))).toEqual({
      "grid-template-columns": "auto 1fr",
    });
    expect(declarations(recipe("automation-summary-columns"))).toEqual({
      "grid-template-columns": "auto minmax(0,1fr)",
    });
    const runDetails = recipe("sm:run-log-detail-columns");
    expect(declarations(runDetails)).toEqual({
      "grid-template-columns": "5rem minmax(0,1fr)",
    });
    expect(runDetails.parent).toMatchObject({
      type: "atrule",
      name: "media",
      params: "(width>=40rem)",
    });

    const radiusContracts = {
      "rounded-pill": "100px",
      "rounded-composer": "15px",
      "rounded-file-action": "7px",
      "rounded-facet-control": "3px",
    };
    for (const [name, radius] of Object.entries(radiusContracts)) {
      expect(declarations(recipe(name))).toEqual({ "border-radius": radius });
    }

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
