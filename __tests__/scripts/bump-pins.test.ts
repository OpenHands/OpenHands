// @vitest-environment node
//
// scripts/bump-pins.mjs writes the upstream release pins that the bump-pins
// workflow resolved. These tests run it on a small copy of the files it edits,
// and once on this repository itself so the script cannot drift from the
// layout it has to edit.
import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import {
  applyPins,
  EXAMPLE_FILES,
  replaceVersion,
  setLockedPackage,
  setPin,
} from "../../scripts/bump-pins.mjs";

const repoRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);
const json = (value: unknown) => `${JSON.stringify(value, null, 2)}\n`;

const CLIENT = "@openhands/typescript-client";
const EXTENSIONS = "@openhands/extensions";
const tarball = (name: string, version: string) =>
  `https://registry.npmjs.org/${name}/-/${name.split("/")[1]}-${version}.tgz`;

// Registry manifests of the releases the tests bump to.
function manifest(name: string, version: string, extra: object = {}) {
  const dependencies =
    name === CLIENT
      ? { ws: "^8.20.0", "@openrouter/sdk": "^1.2.11" }
      : undefined;
  const peerDependencies =
    name === EXTENSIONS ? { react: ">=18.0.0" } : undefined;
  return {
    name,
    version,
    dependencies,
    peerDependencies,
    dist: { tarball: tarball(name, version), integrity: `sha512-${version}` },
    ...extra,
  };
}

const FILES: Record<string, string> = {
  "config/defaults.json": json({
    versions: {
      agentServer: "1.61.0",
      agentCanvas: "1.38.0",
      automation: "1.31.0",
    },
    images: { agentServer: "ghcr.io/openhands/agent-server" },
    ports: { agentServer: 18000, automation: 18001 },
    packages: {
      agentServer: "openhands-agent-server",
      automation: "openhands-automation",
    },
  }),
  "package.json": json({
    name: "demo",
    dependencies: {
      [EXTENSIONS]: "0.41.0",
      [CLIENT]: "1.61.0",
      react: "19.0.0",
    },
  }),
  "package-lock.json": json({
    name: "demo",
    lockfileVersion: 3,
    packages: {
      "": {
        name: "demo",
        dependencies: {
          [EXTENSIONS]: "0.41.0",
          [CLIENT]: "1.61.0",
          react: "19.0.0",
        },
      },
      [`node_modules/${EXTENSIONS}`]: {
        version: "0.41.0",
        resolved: tarball(EXTENSIONS, "0.41.0"),
        integrity: "sha512-0.41.0",
        license: "MIT",
        engines: { node: ">=18.20.0" },
        peerDependencies: { react: ">=18.0.0" },
      },
      [`node_modules/${CLIENT}`]: {
        version: "1.61.0",
        resolved: tarball(CLIENT, "1.61.0"),
        integrity: "sha512-1.61.0",
        license: "MIT",
        dependencies: { "@openrouter/sdk": "^1.2.11", ws: "^8.20.0" },
      },
      "node_modules/react": { version: "19.0.0", integrity: "sha512-react" },
    },
  }),
  ".agents/skills/local-stack-runtime/references/guide.md":
    'Default: released PyPI version `1.61.0` (e.g., "1.61.0")\n',
  "scripts/dev-safe.mjs": ' * - OH_AGENT_SERVER_VERSION: e.g., "1.61.0"\n',
  "scripts/check-sdk-version-sync.mjs": ' *   "openhands-sdk>=1.61.0,<2.0.0"\n',
  "__tests__/scripts/dev-safe.test.ts":
    'expect(cmd.source).toBe("PyPI (1.61.0, default)"); // not 11.61.0\n',
};

describe("setPin", () => {
  it("sets the version-valued entry and leaves same-named keys alone", () => {
    const next = setPin(FILES["config/defaults.json"], "agentServer", "1.62.0");
    expect(JSON.parse(next)).toEqual({
      ...JSON.parse(FILES["config/defaults.json"]),
      versions: {
        agentServer: "1.62.0",
        agentCanvas: "1.38.0",
        automation: "1.31.0",
      },
    });
  });

  it("handles scoped package names", () => {
    const next = setPin(FILES["package.json"], CLIENT, "1.62.0");
    expect(JSON.parse(next).dependencies).toEqual({
      [EXTENSIONS]: "0.41.0",
      [CLIENT]: "1.62.0",
      react: "19.0.0",
    });
  });

  it("refuses a text without exactly one such pin", () => {
    expect(() => setPin("{}", "agentServer", "1.62.0")).toThrow(
      'expected one "agentServer": "X.Y.Z" pin, found 0',
    );
    expect(() =>
      setPin('{"a": {"x": "1.0.0"}, "b": {"x": "2.0.0"}}', "x", "3.0.0"),
    ).toThrow('expected one "x": "X.Y.Z" pin, found 2');
    // A range is not a pin this script may rewrite.
    expect(() => setPin('{"x": "^1.0.0"}', "x", "3.0.0")).toThrow("found 0");
  });
});

describe("replaceVersion", () => {
  it("replaces the version only where it stands alone", () => {
    expect(
      replaceVersion(
        '"1.61.0", ==1.61.0, (>=1.61.0), `1.61.0` but 11.61.0, 1.61.01 and 2.1.61.0',
        "1.61.0",
        "1.62.0",
      ),
    ).toBe(
      '"1.62.0", ==1.62.0, (>=1.62.0), `1.62.0` but 11.61.0, 1.61.01 and 2.1.61.0',
    );
  });
});

describe("setLockedPackage", () => {
  it("moves the root pin and the package entry, and nothing else", () => {
    const next = setLockedPackage(
      FILES["package-lock.json"],
      CLIENT,
      manifest(CLIENT, "1.62.0"),
    );
    const before = JSON.parse(FILES["package-lock.json"]);
    before.packages[""].dependencies[CLIENT] = "1.62.0";
    Object.assign(before.packages[`node_modules/${CLIENT}`], {
      version: "1.62.0",
      resolved: tarball(CLIENT, "1.62.0"),
      integrity: "sha512-1.62.0",
    });
    expect(next).toBe(json(before));
  });

  it("stops when the release changes the dependencies the lockfile records", () => {
    for (const [name, extra, field] of [
      [
        CLIENT,
        { dependencies: { ws: "^9.0.0", "@openrouter/sdk": "^1.2.11" } },
        "dependencies",
      ],
      [CLIENT, { dependencies: { ws: "^8.20.0" } }, "dependencies"],
      [
        CLIENT,
        { optionalDependencies: { fsevents: "^2.0.0" } },
        "optionalDependencies",
      ],
      [
        EXTENSIONS,
        { peerDependencies: { react: ">=19.0.0" } },
        "peerDependencies",
      ],
      [
        EXTENSIONS,
        { peerDependenciesMeta: { react: { optional: true } } },
        "peerDependenciesMeta",
      ],
    ] as const) {
      expect(() =>
        setLockedPackage(
          FILES["package-lock.json"],
          name,
          manifest(name, "9.9.9", extra),
        ),
      ).toThrow(`${name}@9.9.9 changes its ${field}`);
    }
  });

  it("refuses a package that is not a locked direct dependency", () => {
    expect(() =>
      setLockedPackage(
        FILES["package-lock.json"],
        "left-pad",
        manifest("left-pad", "1.0.0"),
      ),
    ).toThrow("left-pad is not a locked direct dependency");
  });
});

describe("applyPins", () => {
  let root: string;
  let requested: string[];
  const read = (rel: string) => readFileSync(path.join(root, rel), "utf-8");
  const getManifest = async (name: string, version: string) => {
    requested.push(`${name}@${version}`);
    return manifest(name, version);
  };

  beforeEach(() => {
    root = mkdtempSync(path.join(tmpdir(), "bump-pins-"));
    requested = [];
    for (const [rel, text] of Object.entries(FILES)) {
      mkdirSync(path.dirname(path.join(root, rel)), { recursive: true });
      writeFileSync(path.join(root, rel), text);
    }
  });

  afterEach(() => {
    rmSync(root, { recursive: true, force: true });
  });

  it("writes every pin, the lockfile entries and the version examples", async () => {
    const changed = await applyPins(
      root,
      { sdk: "1.62.0", automation: "1.32.1", extensions: "0.42.0" },
      getManifest,
    );

    expect(changed.sort()).toEqual(Object.keys(FILES).sort());
    expect(JSON.parse(read("config/defaults.json")).versions).toEqual({
      agentServer: "1.62.0",
      agentCanvas: "1.38.0",
      automation: "1.32.1",
    });
    expect(JSON.parse(read("package.json")).dependencies).toEqual({
      [EXTENSIONS]: "0.42.0",
      [CLIENT]: "1.62.0",
      react: "19.0.0",
    });
    const lock = JSON.parse(read("package-lock.json"));
    expect(lock.packages[""].dependencies).toEqual({
      [EXTENSIONS]: "0.42.0",
      [CLIENT]: "1.62.0",
      react: "19.0.0",
    });
    expect(lock.packages[`node_modules/${EXTENSIONS}`]).toMatchObject({
      version: "0.42.0",
      resolved: tarball(EXTENSIONS, "0.42.0"),
      integrity: "sha512-0.42.0",
    });
    expect(lock.packages[`node_modules/${CLIENT}`].version).toBe("1.62.0");
    for (const rel of EXAMPLE_FILES) {
      expect(read(rel)).toBe(
        FILES[rel].replaceAll("1.61.0", "1.62.0").replace("11.62.0", "11.61.0"),
      );
    }
    expect(requested.sort()).toEqual([
      `${EXTENSIONS}@0.42.0`,
      `${CLIENT}@1.62.0`,
    ]);
  });

  it("changes nothing, and asks the registry nothing, once the pins are in place", async () => {
    const pins = { sdk: "1.62.0", automation: "1.32.1", extensions: "0.42.0" };
    await applyPins(root, pins, getManifest);
    requested = [];

    expect(await applyPins(root, pins, getManifest)).toEqual([]);
    expect(requested).toEqual([]);
  });

  it("touches only what a single pin needs", async () => {
    expect(
      await applyPins(
        root,
        { sdk: "1.61.0", automation: "1.31.1", extensions: "0.41.0" },
        getManifest,
      ),
    ).toEqual(["config/defaults.json"]);
    expect(
      (
        await applyPins(
          root,
          { sdk: "1.61.0", automation: "1.31.1", extensions: "0.42.0" },
          getManifest,
        )
      ).sort(),
    ).toEqual(["package-lock.json", "package.json"]);
    expect(requested).toEqual([`${EXTENSIONS}@0.42.0`]);
  });

  it("leaves the checkout untouched when it refuses a release", async () => {
    await expect(
      applyPins(
        root,
        { sdk: "1.62.0", automation: "1.32.1", extensions: "0.42.0" },
        async (name: string, version: string) =>
          manifest(
            name,
            version,
            name === CLIENT ? { dependencies: { ws: "^9.0.0" } } : {},
          ),
      ),
    ).rejects.toThrow(`${CLIENT}@1.62.0 changes its dependencies`);
    for (const [rel, text] of Object.entries(FILES))
      expect(read(rel)).toBe(text);
  });
});

describe("this repository", () => {
  const config = JSON.parse(
    readFileSync(path.join(repoRoot, "config/defaults.json"), "utf-8"),
  ) as { versions: { agentServer: string; automation: string } };
  const pkg = JSON.parse(
    readFileSync(path.join(repoRoot, "package.json"), "utf-8"),
  ) as { dependencies: Record<string, string> };

  it("quotes the agent-server version in every example file the script rewrites", () => {
    for (const rel of EXAMPLE_FILES) {
      const text = readFileSync(path.join(repoRoot, rel), "utf-8");
      expect(
        replaceVersion(text, config.versions.agentServer, "0.0.0"),
        rel,
      ).not.toBe(text);
    }
  });

  it("is left unchanged by its own current pins", async () => {
    // Fails when a pin moves, is duplicated or stops being an exact version.
    const changed = await applyPins(
      repoRoot,
      {
        sdk: config.versions.agentServer,
        automation: config.versions.automation,
        extensions: pkg.dependencies[EXTENSIONS],
      },
      async () => {
        throw new Error("the lockfile does not match package.json");
      },
    );
    expect(changed).toEqual([]);
  });
});
