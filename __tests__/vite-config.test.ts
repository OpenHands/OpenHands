// @vitest-environment node
import viteConfig from "../vite.config";
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

    expect(appBuild.build?.rolldownOptions?.output?.codeSplitting?.groups).toEqual(
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
    expect(config.build?.rollupOptions?.external).toEqual(
      expect.arrayContaining(["react", "react-dom", "react-router"]),
    );
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
});
