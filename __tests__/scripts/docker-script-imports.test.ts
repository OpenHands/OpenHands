// @vitest-environment node
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const repoRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);
const dockerfile = readFileSync(
  path.join(repoRoot, "docker/Dockerfile"),
  "utf-8",
);
const entrypoint = readFileSync(
  path.join(repoRoot, "docker/entrypoint.sh"),
  "utf-8",
);

// The agent-server reads this package from disk, so no import points at it and
// a missing copy only shows up as an agent with no public skills.
describe("Docker image extensions package", () => {
  it("copies the package to the path the entrypoint exports as EXTENSIONS_PATH", () => {
    const exported = entrypoint.match(
      /^export EXTENSIONS_PATH="\$\{EXTENSIONS_PATH:-([^}]+)\}"$/m,
    )?.[1];

    expect(exported).toBe(
      "/opt/agent-canvas/node_modules/@openhands/extensions",
    );
    expect(dockerfile).toContain(
      `COPY --from=frontend-build /build/node_modules/@openhands/extensions ${exported}`,
    );
  });
});

// The image copies scripts/ one file at a time, so a new import in a script
// the entrypoint runs must come with a COPY line, or the container crashes on
// start with ERR_MODULE_NOT_FOUND (#18342).
function imageScriptGraph(): { scripts: Set<string>; packages: Set<string> } {
  const scripts = new Set<string>();
  const packages = new Set<string>();
  const queue = [
    ...entrypoint.matchAll(/\bnode \/opt\/agent-canvas\/([\w.-]+\.mjs)/g),
  ].map((match) => match[1]);

  while (queue.length > 0) {
    const script = queue.pop()!;
    if (scripts.has(script)) continue;
    scripts.add(script);

    // Drop block comments so JSDoc `import("electron")` types don't count.
    const source = readFileSync(
      path.join(repoRoot, "scripts", script),
      "utf-8",
    ).replace(/\/\*[\s\S]*?\*\//g, "");
    const specifiers = [
      ...source.matchAll(/^\s*(?:import|export)\b[^;]*?\bfrom\s+"([^"]+)"/gm),
      ...source.matchAll(/^\s*import\s+"([^"]+)"/gm),
      ...source.matchAll(/\bimport\(\s*"([^"]+)"\s*\)/g),
    ].map((match) => match[1]);

    for (const specifier of specifiers) {
      if (specifier.startsWith("./")) {
        queue.push(specifier.slice(2));
      } else if (!specifier.startsWith("node:")) {
        const [scope, name] = specifier.split("/");
        packages.add(scope.startsWith("@") ? `${scope}/${name}` : scope);
      }
    }
  }

  return { scripts, packages };
}

describe("Docker image script imports", () => {
  const { scripts, packages } = imageScriptGraph();

  it("finds the scripts the entrypoint runs", () => {
    expect(scripts).toContain("static-server.mjs");
    expect(scripts).toContain("runtime-services-info.mjs");
  });

  it.each([...scripts].sort())("copies %s into /opt/agent-canvas", (script) => {
    expect(dockerfile).toContain(
      `COPY scripts/${script} /opt/agent-canvas/${script}`,
    );
  });

  it.each([...packages].sort())(
    "copies the %s package into /opt/agent-canvas/node_modules",
    (pkg) => {
      expect(dockerfile).toContain(
        `COPY --from=frontend-build /build/node_modules/${pkg} /opt/agent-canvas/node_modules/${pkg}`,
      );
    },
  );
});
