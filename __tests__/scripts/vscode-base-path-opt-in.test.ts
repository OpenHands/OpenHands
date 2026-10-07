// @vitest-environment node
//
// Guards the pairing between advertising the editor prefix and serving it.
//
// `OH_VSCODE_BASE_PATH` changes what `/api/vscode/url` advertises: agent-server
// appends the prefix to the browser origin the frontend sends. Nothing about
// setting it makes that URL resolve — the origin has to route the prefix to the
// editor port as well. A launcher that sets it without registering the route
// advertises `<origin>/vscode/…`, which serves the canvas SPA shell, so the
// editor button opens a second copy of the canvas.
//
// `buildAgentServerEnv` is shared by every launcher, so when the prefix was a
// field on its config object it was on for all of them while only some had the
// route. It is now an explicit argument, and this file asserts that no launcher
// passes it without also serving it.
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import {
  buildAgentServerEnv,
  buildSafeDevConfig,
} from "../../scripts/dev-safe.mjs";

const repoRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);

function readScript(name: string): string {
  return readFileSync(path.join(repoRoot, "scripts", name), "utf-8");
}

/**
 * Source text of every `buildAgentServerEnv(...)` call in `source`, with
 * balanced parentheses so nested calls and object literals are included.
 */
function buildAgentServerEnvCalls(source: string): string[] {
  const calls: string[] = [];
  const needle = "buildAgentServerEnv(";
  let from = 0;

  for (;;) {
    const start = source.indexOf(needle, from);
    if (start === -1) break;
    from = start + needle.length;

    // Skip the definition and the import/export lists, which are not calls.
    const lineStart = source.lastIndexOf("\n", start) + 1;
    const line = source.slice(lineStart, start);
    if (line.includes("function ")) continue;

    let depth = 0;
    let end = start + needle.length - 1;
    for (let i = end; i < source.length; i += 1) {
      if (source[i] === "(") depth += 1;
      else if (source[i] === ")") {
        depth -= 1;
        if (depth === 0) {
          end = i;
          break;
        }
      }
    }
    calls.push(source.slice(start, end + 1));
  }

  return calls;
}

function optsIntoPrefixMode(source: string): boolean {
  return buildAgentServerEnvCalls(source).some((call) =>
    call.includes("vscodeBasePath"),
  );
}

describe("editor base path is opt-in", () => {
  it("omits OH_VSCODE_BASE_PATH unless the caller asks for it", () => {
    const config = buildSafeDevConfig(process.cwd(), {
      OH_SESSION_API_KEY_PATH: path.join(repoRoot, "node_modules", ".test-key"),
    });

    const env = buildAgentServerEnv(config);

    // Not "" or undefined-but-present: agent-server reads the variable's
    // presence, so an empty string would still put it into prefix-mode.
    expect(env).not.toHaveProperty("OH_VSCODE_BASE_PATH");
    // The port is unconditional — the editor still runs, it is just advertised
    // on its own port rather than under a prefix.
    expect(env.OH_VSCODE_PORT).toBe(String(config.vscodePort));
  });

  it("sets OH_VSCODE_BASE_PATH to exactly what the caller passed", () => {
    const config = buildSafeDevConfig(process.cwd(), {
      OH_SESSION_API_KEY_PATH: path.join(repoRoot, "node_modules", ".test-key"),
    });

    const env = buildAgentServerEnv(config, { vscodeBasePath: "/editor" });

    expect(env.OH_VSCODE_BASE_PATH).toBe("/editor");
  });
});

describe("every launcher that advertises the prefix also serves it", () => {
  it("dev-with-automation opts in and routes through getLocalServiceRoutes", () => {
    const source = readScript("dev-with-automation.mjs");
    expect(optsIntoPrefixMode(source)).toBe(true);
    // The route lives in getLocalServiceRoutes, which both the static server
    // and the ingress build their tables from. Behaviour is asserted directly
    // in dev-with-automation.test.ts.
    expect(source).toContain("function getLocalServiceRoutes");
    // IPv4 loopback, not `localhost`: these services bind to 127.0.0.1 and
    // `localhost` can resolve to ::1, which is why every other local proxy
    // target in this repo is written the same way.
    expect(source).toMatch(
      /config\.vscodeBasePath,\s*\n\s*`http:\/\/127\.0\.0\.1/,
    );
  });

  it("dev-static opts in and reuses the same route table", () => {
    const source = readScript("dev-static.mjs");
    expect(optsIntoPrefixMode(source)).toBe(true);
    // Both of its proxies (static server and ingress) must build their routes
    // from the shared helper rather than a hand-maintained copy. The copy this
    // replaced had already drifted: it was missing the editor prefix.
    expect(source).toMatch(
      /function buildLocalServiceRouteArgs[\s\S]*?buildRouteArgs\(\s*getLocalServiceRoutes\(/,
    );
    const routeArgUses = source.match(
      /\.\.\.buildLocalServiceRouteArgs\(config\)/g,
    );
    expect(routeArgUses).toHaveLength(2);
  });

  it("dev:minimal opts in and proxies the prefix through Vite", () => {
    // This mode runs agent-server and Vite with nothing in front of them, so
    // Vite's own proxy is the only thing that can serve the prefix on the
    // origin the browser is on.
    const source = readScript("dev-safe.mjs");
    expect(optsIntoPrefixMode(source)).toBe(true);
    expect(source).toContain("VITE_VSCODE_BASE_PATH: config.vscodeBasePath");
    expect(source).toContain("VITE_VSCODE_TARGET:");

    const viteConfig = readFileSync(
      path.join(repoRoot, "vite.config.ts"),
      "utf-8",
    );
    expect(viteConfig).toContain("VITE_VSCODE_BASE_PATH");
    // The editor upgrades to a WebSocket as soon as the workbench loads, so
    // the proxy entry has to carry ws.
    expect(viteConfig).toMatch(
      /\[VITE_VSCODE_BASE_PATH\]: \{[^}]*target: VITE_VSCODE_TARGET,[^}]*ws: true,/s,
    );
  });

  it("dev-extra-backend stays out of prefix-mode", () => {
    // Its browser origin belongs to a different stack, so a prefix there either
    // does not resolve or resolves to the bundled stack's editor — handing back
    // another container's workspace. No global prefix can disambiguate them.
    //
    // Opting out is not by itself what hides the control: this launcher still
    // starts the editor, so its /api/vscode/status reports it available. What
    // hides it is that with no prefix configured, agent-server appends nothing
    // to the origin and the URL comes back as the canvas root — which
    // `isVSCodeUrlServedByOrigin` rejects. That behaviour is asserted in
    // __tests__/hooks/use-unified-vscode-url.test.tsx; this only pins the
    // launcher's half.
    const source = readScript("dev-extra-backend.mjs");
    expect(optsIntoPrefixMode(source)).toBe(false);
  });

  it("advertises the prefix on the servers that inject into the document", () => {
    // Only the static server rewrites index.html, so only it can tell the
    // frontend what this origin serves; the ingress in front of it routes the
    // same prefix but proxies the document through untouched. Passing the flag
    // to the ingress would also be a hard error — it does not accept it.
    for (const script of ["dev-with-automation.mjs", "dev-static.mjs"]) {
      const source = readScript(script);
      const advertises = source.match(
        /\.\.\.getVSCodeAdvertiseArgs\(config\)/g,
      );
      expect(advertises, `${script} advertises exactly once`).toHaveLength(1);
    }

    // Vite serves the document in full-stack dev mode, so the advertisement is
    // an env var there rather than a server flag.
    expect(readScript("dev-with-automation.mjs")).toContain(
      "viteEnv.VITE_VSCODE_BASE_PATH = config.vscodeBasePath",
    );
  });

  it("full-stack dev advertises the prefix only where Vite also proxies it", () => {
    // This stack has two supported browser origins: the ingress, and Vite's own
    // port — the latter is in AUTOMATION_CORS_ORIGINS precisely so it can be
    // browsed directly. The ingress routes the prefix itself, but on the Vite
    // origin only Vite's proxy can, and vite.config.ts registers that proxy
    // only when it has a target as well as a prefix. Advertising the prefix
    // without the target would put a visible button on the Vite origin whose
    // URL falls through to the SPA — the dead button this gating exists to
    // prevent. So the two env vars have to be set together, in one block.
    const source = readScript("dev-with-automation.mjs");
    // Matched to the block's closing brace at its own indent, so a `${...}`
    // inside the body does not end the match early.
    const block = source.match(
      /if \(config\.launchAgentServer && config\.vscodeBasePath\) \{\n([\s\S]*?VITE_VSCODE[\s\S]*?)\n {2}\}/,
    );
    expect(
      block,
      "the viteEnv editor block is still recognizable",
    ).not.toBeNull();
    expect(block?.[1]).toContain(
      "VITE_VSCODE_BASE_PATH = config.vscodeBasePath",
    );
    // The editor is its own process on its own port, so the proxy target is
    // that port and not the backend/ingress host.
    expect(block?.[1]).toContain(
      "VITE_VSCODE_TARGET = `http://127.0.0.1:${config.vscodePort}`",
    );
  });

  it("gates advertising on exactly the condition that adds the route", () => {
    // If these two guards ever disagree, one of the two failure modes returns:
    // an advertised prefix with no route (control opens the SPA), or a routed
    // prefix nobody advertises (feature silently off). static-server rejects
    // the first at startup; this pins the source of both.
    const source = readScript("dev-with-automation.mjs");
    const advertiseGuard = source.match(
      /function getVSCodeAdvertiseArgs\(config\) \{\s*if \(([^)]*)\) return \[\];/,
    );
    const referrerGuard = source.match(
      /function getNoReferrerPrefixArgs\(config\) \{\s*if \(([^)]*)\) return \[\];/,
    );
    expect(advertiseGuard?.[1]).toBe(
      "!config.launchAgentServer || !config.vscodeBasePath",
    );
    expect(advertiseGuard?.[1]).toBe(referrerGuard?.[1]);
  });
});
