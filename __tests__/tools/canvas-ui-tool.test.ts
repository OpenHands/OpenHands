// @vitest-environment node

import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import {
  CANVAS_UI_CLIENT_ACTION_KIND,
  CANVAS_UI_CLIENT_TOOL,
  CANVAS_UI_CLIENT_TOOL_NAME,
  LEGACY_CANVAS_UI_TOOL_NAME,
} from "#/api/canvas-ui-client-tool";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const legacyToolSource = readFileSync(
  resolve(repoRoot, "tools/canvas_ui_tool.py"),
  "utf8",
);
const dockerEntrypoint = readFileSync(
  resolve(repoRoot, "docker/entrypoint.sh"),
  "utf8",
);
const developmentDoc = readFileSync(
  resolve(repoRoot, "docs/DEVELOPMENT.md"),
  "utf8",
);
const sdkVersion = (
  JSON.parse(
    readFileSync(resolve(repoRoot, "config/defaults.json"), "utf8"),
  ) as { versions: { agentServer: string } }
).versions.agentServer;
const pythonFactoryTests = resolve(
  repoRoot,
  "__tests__/tools/test_canvas_ui_tool.py",
);
const uvAvailable =
  spawnSync("uv", ["--version"], { encoding: "utf8" }).status === 0;

describe("canvas_ui client tool", () => {
  it("tells the agent to capture a browser screenshot before opening the browser tab", () => {
    const captureInstruction = "browser_get_state(include_screenshot=true)";
    const openBrowserInstruction = 'command="open_tab", tab="browser"';

    expect(CANVAS_UI_CLIENT_TOOL.description).toContain(captureInstruction);
    expect(CANVAS_UI_CLIENT_TOOL.description).toContain(openBrowserInstruction);
    expect(
      CANVAS_UI_CLIENT_TOOL.description.indexOf(captureInstruction),
    ).toBeLessThan(
      CANVAS_UI_CLIENT_TOOL.description.indexOf(openBrowserInstruction),
    );
  });

  it("exports the semantic tool name and generated action kind", () => {
    expect(CANVAS_UI_CLIENT_TOOL_NAME).toBe("canvas_ui_control");
    expect(CANVAS_UI_CLIENT_ACTION_KIND).toBe("ClientAction_canvas_ui_control");
    expect(CANVAS_UI_CLIENT_TOOL.name).toBe(CANVAS_UI_CLIENT_TOOL_NAME);
    expect(CANVAS_UI_CLIENT_TOOL.name).not.toBe(LEGACY_CANVAS_UI_TOOL_NAME);
  });

  it("retains the Python registration shim for persisted conversations", () => {
    expect(legacyToolSource).toContain("Legacy conversation compatibility");
    expect(legacyToolSource).toContain(
      'register_tool("canvas_ui", CanvasUITool)',
    );
  });

  // Frontend CI (`npm test`) does not install uv. When uv is present, this
  // actually invokes the factory against the pinned SDK instead of inspecting
  // source text. `__tests__/tools/test_canvas_ui_tool.py` is the same suite.
  it.skipIf(!uvAvailable)(
    "invokes the FinishTool factory for leftover response_schema params",
    () => {
      // Preset automations advertise FinishTool with params={response_schema: TaskOutcome}.
      // The builtin FinishTool.create() raises if any params remain, which 500s
      // POST /api/conversations/{id}/events after the other executors initialize.
      const result = spawnSync(
        "uv",
        [
          "run",
          "--with",
          `openhands-sdk==${sdkVersion}`,
          "python",
          pythonFactoryTests,
        ],
        {
          encoding: "utf8",
          cwd: repoRoot,
          env: { ...process.env, OPENHANDS_SUPPRESS_BANNER: "1" },
        },
      );
      expect(result.error).toBeUndefined();
      expect(result.status, `${result.stdout}\n${result.stderr}`).toBe(0);
    },
    120_000,
  );

  it("imports the FinishTool factory at Docker agent-server startup", () => {
    expect(dockerEntrypoint).toContain(
      'AGENT_SERVER_IMPORT_MODULES="canvas_ui_tool"',
    );
    expect(dockerEntrypoint).toContain(
      '--import-modules "$AGENT_SERVER_IMPORT_MODULES"',
    );
    expect(dockerEntrypoint).toContain("OH_EXTRA_PYTHON_PATH");
  });

  it("documents that preset setup.sh lives in OpenHands/automation", () => {
    expect(developmentDoc).toContain("presets/*/setup.sh");
    expect(developmentDoc).toContain("OpenHands/automation");
    expect(developmentDoc).toContain("openhands-agent-server==${SDK_VERSION}");
  });
});
