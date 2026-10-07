import { describe, expect, it } from "vitest";

import {
  type AgentServerInfo,
  getBackendExecutionMode,
} from "#/api/agent-server-compatibility";

// @spec BM-004 — Display the active backend's execution mode
describe("getBackendExecutionMode", () => {
  it("reads the shipped conversation_runtime field", () => {
    // Arrange
    const serverInfo = {
      version: "1.50.0",
      conversation_runtime: "docker",
    } as AgentServerInfo;

    // Act / Assert
    expect(getBackendExecutionMode(serverInfo)).toBe("docker");
  });

  it("prefers execution_runtime when both fields are present", () => {
    // Arrange
    const serverInfo = {
      version: "1.50.0",
      execution_runtime: "local",
      conversation_runtime: "docker",
    } as AgentServerInfo;

    // Act / Assert
    expect(getBackendExecutionMode(serverInfo)).toBe("local");
  });

  it.each([null, undefined])(
    "returns null when the server info is %s",
    (serverInfo) => {
      // Act / Assert
      expect(getBackendExecutionMode(serverInfo)).toBeNull();
    },
  );

  it("returns null when the server omits the runtime field", () => {
    // Arrange
    const serverInfo = { version: "1.44.0" } as AgentServerInfo;

    // Act / Assert
    expect(getBackendExecutionMode(serverInfo)).toBeNull();
  });

  it("returns null for an unrecognized runtime value", () => {
    // Arrange
    const serverInfo = {
      version: "1.50.0",
      conversation_runtime: "podman",
    } as unknown as AgentServerInfo;

    // Act / Assert
    expect(getBackendExecutionMode(serverInfo)).toBeNull();
  });
});
