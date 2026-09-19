// @vitest-environment node
//
// Drift-detection for the Docker install path's editor route.
//
// The VSCode button advertises a URL built by agent-server from
// OH_VSCODE_BASE_PATH, and that URL only resolves because the static server
// carries a route for the same prefix to the same port. Those two facts live in
// separate files (docker/entrypoint.sh, config/defaults.json via the Dockerfile's
// generated defaults.env), so nothing but a test stops them drifting apart and
// leaving a button that points at the canvas shell instead of the editor.
//
// The npm launcher's equivalent wiring is covered in dev-with-automation.test.ts
// against the real functions. This file covers the shell/Docker half: the
// entrypoint has no importable surface, so the env-resolution block is extracted
// between its markers and executed under bash, which exercises the shipped
// precedence rather than asserting that particular strings appear in the file.
import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const repoRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);

function read(rel: string): string {
  return readFileSync(path.join(repoRoot, rel), "utf-8");
}

const defaults = JSON.parse(read("config/defaults.json")) as {
  ports: { vscode: number; proxy: number };
  paths: { vscodeBasePath: string };
};
const entrypoint = read("docker/entrypoint.sh");
const dockerfile = read("docker/Dockerfile");

// Both static-server invocations (the normal one and the --auth-required
// public-mode one started when PUBLIC_MODE_PORT is set) must carry the route;
// the public-mode server is what the auth-mode E2E suite drives.
function staticServerInvocations(): string[] {
  return entrypoint
    .split("node /opt/agent-canvas/static-server.mjs")
    .slice(1)
    .map((chunk) => chunk.split("\nSTATIC_PID")[0].split("\n  PIDS")[0]);
}

// ── Executing the entrypoint's editor-config block ──────────────────────────
// The block resolves the editor port/prefix from the OH_* variables, this
// image's aliases and the generated defaults.env, then exports the pair to
// agent-server and builds the route string the static servers register. Those
// are two consumers of one setting, so the tests below run the real block and
// compare what each consumer ends up seeing.
const BLOCK_START = "# >>> vscode-config";
const BLOCK_END = "# <<< vscode-config";

function editorConfigBlock(): string {
  const start = entrypoint.indexOf(BLOCK_START);
  const end = entrypoint.indexOf(BLOCK_END);
  if (start === -1 || end === -1) {
    throw new Error(
      `docker/entrypoint.sh is missing the "${BLOCK_START}"/"${BLOCK_END}" markers; ` +
        "the editor-config block can no longer be located, so its behavior is untested.",
    );
  }
  return entrypoint.slice(start, end);
}

interface ResolvedEditorConfig {
  status: number | null;
  stderr: string;
  /** What agent-server is told, and therefore what /api/vscode/url advertises. */
  advertisedBasePath: string;
  advertisedPort: string;
  /** What every static-server instance registers. */
  route: string;
}

function resolveEditorConfig(
  env: Record<string, string> = {},
): ResolvedEditorConfig {
  const script = [
    "set -uo pipefail",
    // Defined near the top of entrypoint.sh, above the extracted block.
    `log_error() { printf 'ERROR: %s\\n' "$*" >&2; }`,
    editorConfigBlock(),
    `printf '%s\\n%s\\n%s\\n' "$OH_VSCODE_BASE_PATH" "$OH_VSCODE_PORT" "$VSCODE_ROUTE"`,
  ].join("\n");

  // Deliberately not inheriting the ambient environment: a developer with
  // OH_VSCODE_* exported would otherwise change what these tests measure.
  const res = spawnSync("bash", ["-c", script], {
    encoding: "utf-8",
    env: { PATH: process.env.PATH ?? "", ...env },
  });
  const [advertisedBasePath = "", advertisedPort = "", route = ""] = res.stdout
    .trim()
    .split("\n");
  return {
    status: res.status,
    stderr: res.stderr,
    advertisedBasePath,
    advertisedPort,
    route,
  };
}

/** The invariant: the advertised URL's prefix/port are the ones being routed. */
function expectRouteMatchesAdvertised(resolved: ResolvedEditorConfig): void {
  expect(resolved.status).toBe(0);
  expect(resolved.route).toBe(
    `${resolved.advertisedBasePath}=http://127.0.0.1:${resolved.advertisedPort}`,
  );
}

describe("docker editor route", () => {
  it("centralizes the base path and port in defaults.json", () => {
    expect(defaults.paths.vscodeBasePath).toBe("/vscode");
    expect(defaults.paths.vscodeBasePath.startsWith("/")).toBe(true);
    expect(Number.isInteger(defaults.ports.vscode)).toBe(true);
  });

  it("exports both values from defaults.json into the generated defaults.env", () => {
    // The container has no jq/python, so the Dockerfile bakes defaults.json
    // into a shell-sourceable env file. A value missing here silently falls
    // back to the hardcoded default in entrypoint.sh.
    expect(dockerfile).toContain(
      "'CONFIG_VSCODE_BASE_PATH=' + c.paths.vscodeBasePath",
    );
    expect(dockerfile).toContain("'CONFIG_VSCODE_PORT=' + c.ports.vscode");
  });

  it("registers the editor route on the normal static-server instance", () => {
    const invocations = staticServerInvocations();
    // Normal + public-mode. If this count changes, decide deliberately which
    // of the two the new instance resembles.
    expect(invocations).toHaveLength(2);

    const [normal] = invocations;
    expect(normal).toContain('--route "$VSCODE_ROUTE"');

    // The route string is assigned once, beside the exports it is derived
    // from. Two independently-built route strings are the drift this whole
    // file exists to prevent.
    const assignments = entrypoint.match(/^VSCODE_ROUTE=/gm) ?? [];
    expect(assignments).toHaveLength(1);
  });

  it("advertises the editor prefix on the instance that routes it", () => {
    // Routing the editor and telling the frontend about it are the same
    // decision. static-server refuses to start if the advertised prefix has no
    // route, so this pins the other direction: an instance that routes the
    // editor must also advertise it, or the control never renders and the
    // feature is silently off.
    const [normal] = staticServerInvocations();
    expect(normal).toContain('--vscode-base-path "$VSCODE_BASE_PATH"');
  });

  it("keeps the editor off the public-mode (--auth-required) instance", () => {
    // --auth-required only decides whether the session key is injected into
    // the served HTML: the dispatcher matches routes before consulting it, so
    // it does not gate proxied paths. The other routes are safe on that
    // footing because agent-server checks the session key itself, but the
    // editor's own credential is the connection token in its query string,
    // and agent-server derives that from session_api_keys[0] — the same secret
    // that authenticates /api. Routing it here would publish that secret in a
    // browser-navigable URL on the origin whose whole purpose is to exercise
    // the unauthenticated case.
    const publicMode = staticServerInvocations().find((invocation) =>
      invocation.includes("--auth-required"),
    );
    expect(publicMode).toBeDefined();
    expect(publicMode).not.toContain("VSCODE_ROUTE");
    // And it must not advertise one either. Omitting only the route would
    // leave the control rendering — the agent-server this instance shares with
    // the main one still reports the editor as available — and the click would
    // fall through to the SPA.
    expect(publicMode).not.toContain("--vscode-base-path");
  });

  it("sends Referrer-Policy: no-referrer on the editor path", () => {
    // The advertised URL carries the connection token as a query parameter and
    // the workbench loads webviews, previews and extension content from that
    // document, so a Referer would carry the token to each of them.
    const [normal] = staticServerInvocations();
    expect(normal).toContain('--no-referrer-prefix "$VSCODE_BASE_PATH"');
  });

  it("routes the editor to its own port, not the agent-server", () => {
    // The editor is a separate process. Pointing the prefix at the
    // agent-server port would 404 the workbench.
    expect(entrypoint).toMatch(
      /^VSCODE_ROUTE="\$\{VSCODE_BASE_PATH\}=http:\/\/127\.0\.0\.1:\$\{VSCODE_PORT\}"$/m,
    );
    expect(defaults.ports.vscode).not.toBe(defaults.ports.proxy);
  });

  it("does not publish the editor port", () => {
    // The single-origin shape is the point: the editor is reachable only
    // through the proxy port's path prefix, so it inherits the canvas's
    // auth/ingress posture instead of needing a second exposed port.
    expect(dockerfile).not.toMatch(
      new RegExp(`^\\s*EXPOSE\\s+${defaults.ports.vscode}\\b`, "m"),
    );
  });
});

// The entrypoint only ever runs inside the Linux image; bash is not a given on a
// Windows developer machine, and CI runs the unit suite on ubuntu only.
describe.skipIf(process.platform === "win32")(
  "docker editor config resolution",
  () => {
    it("advertises and routes the same pair with no overrides", () => {
      const resolved = resolveEditorConfig();
      expectRouteMatchesAdvertised(resolved);
      // Literal fallbacks used when defaults.env is absent — they must not
      // drift from the central config either.
      expect(resolved.advertisedBasePath).toBe(defaults.paths.vscodeBasePath);
      expect(resolved.advertisedPort).toBe(String(defaults.ports.vscode));
    });

    it("takes the defaults baked into defaults.env", () => {
      const resolved = resolveEditorConfig({
        CONFIG_VSCODE_BASE_PATH: "/editor",
        CONFIG_VSCODE_PORT: "9001",
      });
      expectRouteMatchesAdvertised(resolved);
      expect(resolved.advertisedBasePath).toBe("/editor");
      expect(resolved.advertisedPort).toBe("9001");
    });

    // The regression this block was restructured for: agent-server's own
    // documented variables are what a self-hosted deployment is most likely to
    // already set, and setting one of them used to move the editor without
    // moving the route.
    it("moves the route when only OH_VSCODE_BASE_PATH is set", () => {
      const resolved = resolveEditorConfig({
        OH_VSCODE_BASE_PATH: "/editor",
        CONFIG_VSCODE_BASE_PATH: "/vscode",
      });
      expectRouteMatchesAdvertised(resolved);
      expect(resolved.advertisedBasePath).toBe("/editor");
      expect(resolved.route).toContain("/editor=");
    });

    it("moves the route when only OH_VSCODE_PORT is set", () => {
      const resolved = resolveEditorConfig({
        OH_VSCODE_PORT: "9001",
        CONFIG_VSCODE_PORT: "8001",
      });
      expectRouteMatchesAdvertised(resolved);
      expect(resolved.advertisedPort).toBe("9001");
      expect(resolved.route).toBe("/vscode=http://127.0.0.1:9001");
    });

    it("honours this image's aliases too", () => {
      const resolved = resolveEditorConfig({
        VSCODE_BASE_PATH: "/editor",
        VSCODE_PORT: "9001",
      });
      expectRouteMatchesAdvertised(resolved);
      expect(resolved.advertisedBasePath).toBe("/editor");
      expect(resolved.advertisedPort).toBe("9001");
    });

    it("keeps one effective pair when both names are set and disagree", () => {
      const resolved = resolveEditorConfig({
        OH_VSCODE_BASE_PATH: "/editor",
        OH_VSCODE_PORT: "9001",
        VSCODE_BASE_PATH: "/vscode",
        VSCODE_PORT: "8001",
      });
      // Whichever wins, the two consumers must not disagree — and the OH_*
      // variables win, since they are what agent-server itself documents.
      expectRouteMatchesAdvertised(resolved);
      expect(resolved.advertisedBasePath).toBe("/editor");
      expect(resolved.advertisedPort).toBe("9001");
    });

    it.each(["editor", "/editor", "/editor/", "//editor//"])(
      "normalizes %j to one spelling for both consumers",
      (given) => {
        const resolved = resolveEditorConfig({ OH_VSCODE_BASE_PATH: given });
        expectRouteMatchesAdvertised(resolved);
        expect(resolved.advertisedBasePath).toBe("/editor");
      },
    );

    it("refuses a base path that resolves to the site root", () => {
      // Routing "/" to the editor would hand it the whole origin, including the
      // canvas itself — fail loudly at startup instead of serving that.
      const resolved = resolveEditorConfig({ OH_VSCODE_BASE_PATH: "/" });
      expect(resolved.status).not.toBe(0);
      expect(resolved.stderr).toContain("site root");
    });

    // static-server keys its route table by prefix and the editor route is
    // registered last, so a colliding prefix silently *replaces* the earlier
    // route instead of failing. `/api` is the dangerous one: every API call
    // would be proxied to the editor port, which reads as a total outage with
    // no error to explain it. The "/" guard above does not catch these.
    it.each([
      "/api",
      "/sockets",
      "/server_info",
      "/health",
      "/openapi.json",
      "/canvas",
    ])("refuses %j, which would take over an existing route", (given) => {
      const resolved = resolveEditorConfig({
        AGENT_CANVAS_BASE_PATH: "/canvas",
        OH_VSCODE_BASE_PATH: given,
      });
      expect(resolved.status).not.toBe(0);
      expect(resolved.stderr).toContain("collides");
    });

    // The collision guard compares two prefixes, so both have to be normalized
    // the same way or a noncanonical spelling walks straight past it.
    // static-server normalizes whatever `--base-path` it is handed, so every
    // spelling below mounts the canvas at `/canvas` — the guard has to be
    // looking at the same value the router will.
    it.each(["canvas", "/canvas", "/canvas/", "//canvas//"])(
      "refuses an editor prefix colliding with AGENT_CANVAS_BASE_PATH spelled %j",
      (canvasBasePath) => {
        const resolved = resolveEditorConfig({
          AGENT_CANVAS_BASE_PATH: canvasBasePath,
          OH_VSCODE_BASE_PATH: "/canvas",
        });
        expect(resolved.status).not.toBe(0);
        expect(resolved.stderr).toContain("collides");
      },
    );

    it("guards the default canvas mount without being told it", () => {
      // Resolving AGENT_CANVAS_BASE_PATH inside the extracted block is what
      // makes this reachable: a deployment that moves only the editor onto the
      // canvas's default mount never sets AGENT_CANVAS_BASE_PATH at all.
      const resolved = resolveEditorConfig({ OH_VSCODE_BASE_PATH: "/canvas" });
      expect(resolved.status).not.toBe(0);
      expect(resolved.stderr).toContain("collides");
    });

    it("accepts a noncanonical canvas mount that does not collide", () => {
      // The guard must reject overlap, not coexistence: normalizing both sides
      // must not start rejecting layouts that are actually fine.
      const resolved = resolveEditorConfig({
        AGENT_CANVAS_BASE_PATH: "canvas/",
        OH_VSCODE_BASE_PATH: "editor",
      });
      expect(resolved.status).toBe(0);
      expectRouteMatchesAdvertised(resolved);
      expect(resolved.advertisedBasePath).toBe("/editor");
    });

    it.each([
      // static-server's --route parser cuts at the *first* '=', so this parses
      // as prefix "/vs" pointing at the garbage url "code=http://…" — a silent
      // outage under /vs rather than a startup failure.
      ["/vs=code", "may only contain"],
      ["/a b", "may only contain"],
      ["/x?y", "may only contain"],
      ["/x#y", "may only contain"],
      // Multi-segment prefixes are not wrong in principle, but agent-server
      // strips the slashes when building the advertised URL, so the two sides
      // would disagree. Reject rather than silently half-support it.
      ["/deep/path", "single path segment"],
      ["/../api", "single path segment"],
    ])("refuses %j", (given, expectedMessage) => {
      const resolved = resolveEditorConfig({ OH_VSCODE_BASE_PATH: given });
      expect(resolved.status).not.toBe(0);
      expect(resolved.stderr).toContain(expectedMessage);
    });

    it("refuses a non-numeric port", () => {
      // The port is interpolated straight into a proxy target URL, so without
      // this it fails on the first editor request instead of at startup.
      const resolved = resolveEditorConfig({ OH_VSCODE_PORT: "not-a-port" });
      expect(resolved.status).not.toBe(0);
      expect(resolved.stderr).toContain("must be a number");
    });
  },
);
