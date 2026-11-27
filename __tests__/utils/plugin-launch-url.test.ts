import { describe, expect, it } from "vitest";
import type { PluginSpec } from "#/api/conversation-service/agent-server-conversation-service.types";
import { buildPluginLaunchPath } from "#/utils/plugin-launch-url";

describe("buildPluginLaunchPath", () => {
  it("encodes plugins into a /launch path that decodes back to the same specs", () => {
    // Arrange: coordinates whose JSON base64-encodes with URL-special chars (+ / =).
    const plugins: PluginSpec[] = [
      {
        source: "github:OpenHands/extensions",
        ref: "v1.2+3",
        repo_path: "sub/dir",
      },
    ];

    // Act: build the path, then decode the `plugins` param the way /launch does.
    const path = buildPluginLaunchPath(plugins);
    const url = new URL(path, "http://localhost");
    const decoded = JSON.parse(atob(url.searchParams.get("plugins") ?? ""));

    // Assert: it targets /launch and round-trips the coordinates without corruption.
    expect(url.pathname).toBe("/launch");
    expect(decoded).toEqual(plugins);
  });
});
