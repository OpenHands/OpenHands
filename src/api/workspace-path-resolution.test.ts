import { describe, expect, it, vi } from "vitest";

const {
  fileClientConstructorMock,
  getAgentServerClientOptionsMock,
  getHomeMock,
} = vi.hoisted(() => ({
  fileClientConstructorMock: vi.fn(),
  getAgentServerClientOptionsMock: vi.fn(),
  getHomeMock: vi.fn(),
}));

vi.mock("@openhands/typescript-client/clients", () => ({
  FileClient: class {
    constructor(options: unknown) {
      fileClientConstructorMock(options);
    }

    getHome() {
      return getHomeMock();
    }
  },
}));

vi.mock("./agent-server-client-options", () => ({
  getAgentServerClientOptions: getAgentServerClientOptionsMock,
}));

import {
  clearAgentServerHomeDirCache,
  getAgentServerHomeDir,
  resolveAbsoluteAgentServerPath,
} from "./agent-server-home";

const defaultClientOptions = {
  host: "https://agent.example.com",
  apiKey: "agent-key",
  workingDir: "workspace/project",
};

function arrangeHomeLookup(): void {
  clearAgentServerHomeDirCache();
  vi.resetAllMocks();
  getAgentServerClientOptionsMock.mockReturnValue(defaultClientOptions);
}

describe("agent-server workspace path resolution", () => {
  it("looks up the server home with the resolved client options and normalizes its separator", async () => {
    arrangeHomeLookup();
    getHomeMock.mockResolvedValue({ home: "/Users/alice///" });
    const overrides = {
      conversationUrl: "wss://runtime.example.com/conversations/1",
      sessionApiKey: "session-key",
    };

    await expect(getAgentServerHomeDir(overrides)).resolves.toBe(
      "/Users/alice",
    );

    expect(getAgentServerClientOptionsMock).toHaveBeenCalledWith(overrides);
    expect(fileClientConstructorMock).toHaveBeenCalledWith(
      defaultClientOptions,
    );
    expect(getHomeMock).toHaveBeenCalledOnce();
  });

  it("shares one in-flight home lookup between callers for the same host", async () => {
    arrangeHomeLookup();
    getHomeMock.mockResolvedValue({ home: "/home/agent" });

    await expect(
      Promise.all([getAgentServerHomeDir(), getAgentServerHomeDir()]),
    ).resolves.toEqual(["/home/agent", "/home/agent"]);

    expect(getHomeMock).toHaveBeenCalledOnce();
  });

  it("caches home directories independently for each agent-server host", async () => {
    arrangeHomeLookup();
    getAgentServerClientOptionsMock.mockImplementation(
      (overrides: { host?: string } = {}) => ({
        ...defaultClientOptions,
        host: overrides.host ?? defaultClientOptions.host,
      }),
    );
    getHomeMock
      .mockResolvedValueOnce({ home: "/home/first" })
      .mockResolvedValueOnce({ home: "/home/second" });

    await expect(
      Promise.all([
        getAgentServerHomeDir({ host: "https://first.example.com" }),
        getAgentServerHomeDir({ host: "https://second.example.com" }),
      ]),
    ).resolves.toEqual(["/home/first", "/home/second"]);
    await expect(
      getAgentServerHomeDir({ host: "https://first.example.com" }),
    ).resolves.toBe("/home/first");

    expect(getHomeMock).toHaveBeenCalledTimes(2);
  });

  it("retries a home lookup after a transient failure", async () => {
    arrangeHomeLookup();
    getHomeMock
      .mockRejectedValueOnce(new Error("agent server unavailable"))
      .mockResolvedValueOnce({ home: "/home/recovered" });

    await expect(getAgentServerHomeDir()).rejects.toThrow(
      "agent server unavailable",
    );
    await expect(getAgentServerHomeDir()).resolves.toBe("/home/recovered");

    expect(getHomeMock).toHaveBeenCalledTimes(2);
  });

  it.each([undefined, null, "", 0, 42])(
    "rejects an invalid home directory response (%s)",
    async (home) => {
      arrangeHomeLookup();
      getHomeMock.mockResolvedValue({ home });

      await expect(getAgentServerHomeDir()).rejects.toThrow(
        "Agent server returned an empty home directory",
      );
    },
  );

  it("clears a successful cached lookup on request", async () => {
    arrangeHomeLookup();
    getHomeMock
      .mockResolvedValueOnce({ home: "/home/first" })
      .mockResolvedValueOnce({ home: "/home/refreshed" });

    await expect(getAgentServerHomeDir()).resolves.toBe("/home/first");
    clearAgentServerHomeDirCache();
    await expect(getAgentServerHomeDir()).resolves.toBe("/home/refreshed");

    expect(getHomeMock).toHaveBeenCalledTimes(2);
  });

  it.each([
    ["/workspace/project///", "/workspace/project"],
    ["C:\\workspace\\project\\", "C:\\workspace\\project"],
    ["D:/workspace/project///", "D:/workspace/project"],
    ["\\\\server\\share\\", "\\\\server\\share"],
  ])(
    "keeps the absolute path %s without a home lookup",
    async (path, expected) => {
      arrangeHomeLookup();

      await expect(resolveAbsoluteAgentServerPath(path)).resolves.toBe(
        expected,
      );

      expect(getAgentServerClientOptionsMock).not.toHaveBeenCalled();
      expect(getHomeMock).not.toHaveBeenCalled();
    },
  );

  it.each(["", "/", "\\", "///", "\\\\"])(
    "resolves an empty path (%s) to the server home",
    async (path) => {
      arrangeHomeLookup();
      getHomeMock.mockResolvedValue({ home: "/home/agent/" });

      await expect(resolveAbsoluteAgentServerPath(path)).resolves.toBe(
        "/home/agent",
      );

      expect(getHomeMock).toHaveBeenCalledOnce();
    },
  );

  it("anchors a relative workspace below the server home", async () => {
    arrangeHomeLookup();
    getHomeMock.mockResolvedValue({ home: "/Users/alice///" });
    const overrides = {
      host: "https://runtime.example.com/",
      sessionApiKey: "session-key",
    };

    await expect(
      resolveAbsoluteAgentServerPath("workspace/project///", overrides),
    ).resolves.toBe("/Users/alice/workspace/project");

    expect(getAgentServerClientOptionsMock).toHaveBeenCalledWith(overrides);
  });
});
