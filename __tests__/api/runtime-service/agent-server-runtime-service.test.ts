import { FileClient } from "@openhands/typescript-client/clients";
import { RemoteWorkspace } from "@openhands/typescript-client/workspace/remote-workspace";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import AgentServerRuntimeService from "#/api/runtime-service/agent-server-runtime-service";
import { callCloudProxy } from "#/api/cloud/proxy";
import type { Backend } from "#/api/backend-registry/types";

// ─── SDK client mocks ───────────────────────────────────────────────────────

const { executeCommandMock, downloadFileMock } = vi.hoisted(() => ({
  executeCommandMock: vi.fn(),
  downloadFileMock: vi.fn(),
}));

vi.mock("@openhands/typescript-client/workspace/remote-workspace", () => ({
  RemoteWorkspace: vi.fn(function RemoteWorkspaceMock() {
    return { executeCommand: executeCommandMock };
  }),
}));

vi.mock("@openhands/typescript-client/clients", () => ({
  FileClient: vi.fn(function FileClientMock() {
    return { downloadFile: downloadFileMock };
  }),
}));

vi.mock("#/api/agent-server-client-options", () => ({
  getAgentServerClientOptions: vi.fn(() => ({
    host: "http://local-agent.example.com",
    apiKey: "local-key",
    workingDir: "/workspace/project",
  })),
}));

vi.mock("#/api/cloud/proxy", () => ({
  callCloudProxy: vi.fn(),
}));

// ─── Backend fixtures ────────────────────────────────────────────────────────

const cloudBackend: Backend = {
  id: "cloud-1",
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "cloud-api-key",
  kind: "cloud",
};

const CLOUD_CONVERSATION_URL =
  "https://runtime.example.com/api/conversations/conv-1";
const SESSION_KEY = "session-key-abc";

// ─── Helpers ─────────────────────────────────────────────────────────────────

function activateCloud() {
  setRegisteredBackends([cloudBackend]);
  setActiveSelection({ backendId: cloudBackend.id, orgId: null });
}

// ─── Setup ───────────────────────────────────────────────────────────────────

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  vi.mocked(RemoteWorkspace).mockClear();
  vi.mocked(FileClient).mockClear();
  executeCommandMock.mockReset();
  downloadFileMock.mockReset();
  vi.mocked(callCloudProxy).mockReset();
});

afterEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
});

// ─── executeCommand ──────────────────────────────────────────────────────────

describe("AgentServerRuntimeService.executeCommand", () => {
  describe("local backend", () => {
    it("creates RemoteWorkspace with resolved options and delegates", async () => {
      executeCommandMock.mockResolvedValue({
        exit_code: 0,
        stdout: "main\n",
        stderr: "",
      });

      const result = await AgentServerRuntimeService.executeCommand(
        "http://local-agent.example.com/api/conversations/conv-1",
        SESSION_KEY,
        "git rev-parse --abbrev-ref HEAD",
        "/workspace/project",
        10,
      );

      expect(RemoteWorkspace).toHaveBeenCalledTimes(1);
      expect(executeCommandMock).toHaveBeenCalledWith(
        "git rev-parse --abbrev-ref HEAD",
        "/workspace/project",
        10,
      );
      expect(result).toEqual({ exit_code: 0, stdout: "main\n", stderr: "" });
    });

    it("does not call callCloudProxy for local backends", async () => {
      executeCommandMock.mockResolvedValue({
        exit_code: 0,
        stdout: "",
        stderr: "",
      });

      await AgentServerRuntimeService.executeCommand(null, null, "ls", "/", 5);

      expect(callCloudProxy).not.toHaveBeenCalled();
    });
  });

  describe("cloud backend", () => {
    beforeEach(activateCloud);

    it("routes through callCloudProxy with correct path, body, and auth", async () => {
      vi.mocked(callCloudProxy).mockResolvedValue({
        exit_code: 0,
        stdout: "src/index.ts\n",
        stderr: "",
      });

      const result = await AgentServerRuntimeService.executeCommand(
        CLOUD_CONVERSATION_URL,
        SESSION_KEY,
        "find . -type f",
        "/workspace/project",
        30,
      );

      const proxyCall = vi.mocked(callCloudProxy).mock.calls[0][0];
      expect(proxyCall.method).toBe("POST");
      expect(proxyCall.path).toBe("/api/bash/execute_bash_command");
      expect(proxyCall.hostOverride).toBe("https://runtime.example.com");
      expect(proxyCall.body).toEqual({
        command: "find . -type f",
        cwd: "/workspace/project",
        timeout: 30,
      });
      expect(proxyCall.authMode).toBe("session-api-key");
      expect(proxyCall.sessionApiKey).toBe(SESSION_KEY);
      expect(proxyCall.timeoutSeconds).toBe(40);
      expect(result).toEqual({
        exit_code: 0,
        stdout: "src/index.ts\n",
        stderr: "",
      });
    });

    it("omits cwd from proxy body when not provided", async () => {
      vi.mocked(callCloudProxy).mockResolvedValue({ exit_code: 0 });

      await AgentServerRuntimeService.executeCommand(
        CLOUD_CONVERSATION_URL,
        SESSION_KEY,
        "echo hi",
        undefined,
        10,
      );

      const proxyBody = vi.mocked(callCloudProxy).mock.calls[0][0].body as Record<string, unknown>;
      expect(proxyBody).not.toHaveProperty("cwd");
      expect(proxyBody.command).toBe("echo hi");
    });

    it("normalises missing stdout/stderr fields to empty strings", async () => {
      vi.mocked(callCloudProxy).mockResolvedValue({ exit_code: 1 });

      const result = await AgentServerRuntimeService.executeCommand(
        CLOUD_CONVERSATION_URL,
        SESSION_KEY,
        "false",
        undefined,
        5,
      );

      expect(result).toEqual({ exit_code: 1, stdout: "", stderr: "" });
    });

    it("does not create a RemoteWorkspace for cloud calls", async () => {
      vi.mocked(callCloudProxy).mockResolvedValue({ exit_code: 0 });

      await AgentServerRuntimeService.executeCommand(
        CLOUD_CONVERSATION_URL,
        SESSION_KEY,
        "echo ok",
      );

      expect(RemoteWorkspace).not.toHaveBeenCalled();
    });

    it("falls back to local path when conversationUrl is null", async () => {
      executeCommandMock.mockResolvedValue({
        exit_code: 0,
        stdout: "",
        stderr: "",
      });

      await AgentServerRuntimeService.executeCommand(
        null,
        SESSION_KEY,
        "echo ok",
      );

      expect(callCloudProxy).not.toHaveBeenCalled();
      expect(RemoteWorkspace).toHaveBeenCalledTimes(1);
    });
  });
});

// ─── downloadFile ─────────────────────────────────────────────────────────────

describe("AgentServerRuntimeService.downloadFile", () => {
  describe("local backend", () => {
    it("creates FileClient with resolved options and returns the ArrayBuffer", async () => {
      const fileBytes = new TextEncoder().encode("# README");
      downloadFileMock.mockResolvedValue(fileBytes.buffer);

      const result = await AgentServerRuntimeService.downloadFile(
        "http://local-agent.example.com/api/conversations/conv-1",
        SESSION_KEY,
        "/workspace/project/README.md",
      );

      expect(FileClient).toHaveBeenCalledTimes(1);
      expect(downloadFileMock).toHaveBeenCalledWith(
        "/workspace/project/README.md",
      );
      expect(result).toBe(fileBytes.buffer);
    });

    it("does not call callCloudProxy for local backends", async () => {
      downloadFileMock.mockResolvedValue(new ArrayBuffer(0));

      await AgentServerRuntimeService.downloadFile(
        null,
        null,
        "/workspace/file.txt",
      );

      expect(callCloudProxy).not.toHaveBeenCalled();
    });
  });

  describe("cloud backend", () => {
    beforeEach(activateCloud);

    it("routes through callCloudProxy with GET and URL-encoded path", async () => {
      const blob = new Blob([new TextEncoder().encode("file content")]);
      vi.mocked(callCloudProxy).mockResolvedValue(blob);

      const result = await AgentServerRuntimeService.downloadFile(
        CLOUD_CONVERSATION_URL,
        SESSION_KEY,
        "/workspace/project/src/main.ts",
      );

      const proxyCall = vi.mocked(callCloudProxy).mock.calls[0][0];
      expect(proxyCall.method).toBe("GET");
      expect(proxyCall.path).toBe(
        "/api/file/download?path=%2Fworkspace%2Fproject%2Fsrc%2Fmain.ts",
      );
      expect(proxyCall.hostOverride).toBe("https://runtime.example.com");
      expect(proxyCall.authMode).toBe("session-api-key");
      expect(proxyCall.sessionApiKey).toBe(SESSION_KEY);
      expect(proxyCall.responseType).toBe("blob");

      // Blob.arrayBuffer() round-trip: decoded text should match the original.
      expect(new TextDecoder().decode(result)).toBe("file content");
    });

    it("does not create a FileClient for cloud calls", async () => {
      vi.mocked(callCloudProxy).mockResolvedValue(new Blob());

      await AgentServerRuntimeService.downloadFile(
        CLOUD_CONVERSATION_URL,
        SESSION_KEY,
        "/workspace/file.txt",
      );

      expect(FileClient).not.toHaveBeenCalled();
    });

    it("falls back to local path when conversationUrl is null", async () => {
      downloadFileMock.mockResolvedValue(new ArrayBuffer(0));

      await AgentServerRuntimeService.downloadFile(
        null,
        SESSION_KEY,
        "/workspace/file.txt",
      );

      expect(callCloudProxy).not.toHaveBeenCalled();
      expect(FileClient).toHaveBeenCalledTimes(1);
    });
  });
});
