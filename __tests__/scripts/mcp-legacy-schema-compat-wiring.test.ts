// @vitest-environment node
//
// Drift-detection for the mcp legacy-schema reader compatibility added for
// OpenHands/OpenHands#17615.
//
// The agent-server resolves `mcp` 1.x (via the `fastmcp<4` pin) while
// conversations created during the earlier `mcp` 2.x window persisted
// snake_case MCP tool fields. tools/mcp_legacy_schema_compat.py patches the
// strict 1.x reader at import time, so the module must be listed in the
// agent-server's `--import-modules` on every launch path that can restore a
// conversation: the npm/Vite launchers and the Docker entrypoint.
//
// The behavioral coverage lives in tools/tests/test_mcp_legacy_schema_compat.py
// (pytest). This file only pins the wiring, which lives in shell/JS files that
// pytest cannot see.
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import { AGENT_SERVER_IMPORT_MODULES } from "../../scripts/dev-safe.mjs";

const repoRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);

function read(rel: string): string {
  return readFileSync(path.join(repoRoot, rel), "utf-8");
}

const COMPAT_MODULE = "mcp_legacy_schema_compat";

function moduleList(value: string): string[] {
  return value.split(",").map((name) => name.trim());
}

describe("mcp legacy-schema compatibility wiring", () => {
  it("ships the compatibility module beside the other agent-server tools", () => {
    const source = read("tools/mcp_legacy_schema_compat.py");
    expect(source).toContain("def apply_mcp_field_compatibility(");
    // Importing the module is the startup contract, so the call must run at
    // module scope rather than only under an entrypoint.
    expect(source).toMatch(/^apply_mcp_field_compatibility\(\)$/m);
  });

  it("imports the module at agent-server startup in the npm launchers", () => {
    expect(moduleList(AGENT_SERVER_IMPORT_MODULES)).toContain(COMPAT_MODULE);
  });

  it("imports the module at agent-server startup in the Docker entrypoint", () => {
    // Read the assignment, not the `--import-modules` flag, so a value supplied
    // by the environment cannot make this test pass on a stale default.
    const assignment = read("docker/entrypoint.sh").match(
      /^AGENT_SERVER_IMPORT_MODULES="([^"]*)"/m,
    );
    expect(assignment).not.toBeNull();
    expect(moduleList(assignment![1])).toContain(COMPAT_MODULE);
  });

  it("patches before canvas_ui_tool so the SDK builds on patched fields", () => {
    // canvas_ui_tool imports the SDK event graph as an import side effect; the
    // compat module recompiles already-imported models, so ordering it first
    // keeps the patch off the recompile path when nothing has been imported.
    const modules = moduleList(AGENT_SERVER_IMPORT_MODULES);
    const compatIndex = modules.indexOf(COMPAT_MODULE);
    const canvasIndex = modules.indexOf("canvas_ui_tool");
    expect(compatIndex).toBeGreaterThan(-1);
    expect(canvasIndex).toBeGreaterThan(-1);
    expect(compatIndex).toBeLessThan(canvasIndex);
  });

  it("installs the tools directory onto the image's python path", () => {
    // --import-modules resolves bare module names through sys.path, and the
    // Dockerfile exposes tools/ via OH_EXTRA_PYTHON_PATH.
    const entrypoint = read("docker/entrypoint.sh");
    expect(entrypoint).toContain("OH_EXTRA_PYTHON_PATH");
    expect(read("docker/Dockerfile")).toContain("COPY tools/");
  });
});
