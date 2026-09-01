// @vitest-environment node
import { describe, expect, it } from "vitest";

import {
  buildDockerCommand,
  parseArgs,
} from "../../scripts/docker-build.mjs";

describe("docker build target selection", () => {
  it("preserves the legacy final target by default", () => {
    const options = parseArgs([]);
    expect(options.target).toBe("final");
    expect(options.tag).toBe("agent-canvas:local");
  });

  it("selects an immutable amd64 Enterprise image", () => {
    const options = parseArgs(["--enterprise"]);
    expect(options.target).toBe("enterprise-sandbox");
    expect(options.platform).toBe("linux/amd64");
    expect(options.tag).toMatch(
      /^agent-canvas-enterprise-sandbox:\d+\.\d+\.\d+-agent-server-\d+\.\d+\.\d+$/,
    );
  });

  it("builds the selected target with explicit version metadata", () => {
    const options = parseArgs(["--enterprise"]);
    const command = buildDockerCommand(options);

    expect(command).toEqual(
      expect.arrayContaining([
        "--target",
        "enterprise-sandbox",
        "--platform",
        "linux/amd64",
        "--build-arg",
        `AGENT_SERVER_IMAGE=${options.agentServerImage}`,
        "--build-arg",
        `AGENT_SERVER_VERSION=${options.agentServerVersion}`,
      ]),
    );
    expect(command.at(-1)).toBe(".");
  });

  it("supports explicit image, tag, target, and passthrough arguments", () => {
    const options = parseArgs([
      "--target",
      "dev-sandbox",
      "--agent-server-image",
      "registry.example/agent-server:1.45.2-python",
      "--tag",
      "example/dev-sandbox:test",
      "--",
      "--no-cache",
    ]);

    expect(options.agentServerVersion).toBe("1.45.2");
    expect(options.tag).toBe("example/dev-sandbox:test");
    expect(buildDockerCommand(options)).toContain("--no-cache");
  });

  it("rejects unsupported targets", () => {
    expect(() => parseArgs(["--target", "unknown"])).toThrow(
      /Unsupported Docker target/,
    );
  });

  it("rejects non-amd64 Enterprise builds", () => {
    expect(() =>
      parseArgs(["--enterprise", "--platform", "linux/arm64"]),
    ).toThrow(/linux\/amd64/);
  });
});
