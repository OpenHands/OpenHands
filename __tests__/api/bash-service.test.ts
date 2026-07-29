import { BashClient } from "@openhands/typescript-client/clients";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import BashService from "#/api/bash-service/bash-service.api";
import { callCloudProxy } from "#/api/cloud/proxy";
import type { Backend } from "#/api/backend-registry/types";

const { searchEventsMock } = vi.hoisted(() => ({
  searchEventsMock: vi.fn(),
}));

vi.mock("@openhands/typescript-client/clients", () => ({
  BashClient: vi.fn(function BashClientMock() {
    return { searchEvents: searchEventsMock };
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

const localBackend: Backend = {
  id: "local-1",
  name: "Local 1",
  host: "http://local-agent.example.com",
  apiKey: "local-key",
  kind: "local",
};

const cloudBackend: Backend = {
  id: "cloud-1",
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "cloud-api-key",
  kind: "cloud",
};

const CONVERSATION_URL = "https://runtime.example.com/api/conversations/conv-1";
const SESSION_KEY = "session-key-abc";
const BASH_CMD_ID = "cmd-123";

const OUTPUT_1 = {
  id: "out-1",
  timestamp: "2026-01-01T10:00:00.100Z",
  kind: "BashOutput",
  command_id: BASH_CMD_ID,
  order: 0,
  stdout: "hello\n",
  stderr: null,
  exit_code: null,
};

const OUTPUT_2 = {
  id: "out-2",
  timestamp: "2026-01-01T10:00:00.200Z",
  kind: "BashOutput",
  command_id: BASH_CMD_ID,
  order: 1,
  stdout: null,
  stderr: "boom\n",
  exit_code: 1,
};

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  vi.mocked(BashClient).mockClear();
  searchEventsMock.mockReset();
  vi.mocked(callCloudProxy).mockReset();
});

afterEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
});

describe("BashService.listOutputs — local backend", () => {
  beforeEach(() => {
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id, orgId: null });
  });

  it("pages BashOutput events via BashClient with command_id__eq filter", async () => {
    searchEventsMock
      .mockResolvedValueOnce({
        items: [OUTPUT_1],
        next_page_id: "next",
      })
      .mockResolvedValueOnce({ items: [OUTPUT_2] });

    const outputs = await BashService.listOutputs(
      CONVERSATION_URL,
      SESSION_KEY,
      BASH_CMD_ID,
    );

    expect(BashClient).toHaveBeenCalled();
    expect(searchEventsMock).toHaveBeenCalledTimes(2);
    expect(searchEventsMock.mock.calls[0][0]).toMatchObject({
      kind__eq: "BashOutput",
      command_id__eq: BASH_CMD_ID,
      sort_order: "TIMESTAMP",
    });
    expect(searchEventsMock.mock.calls[1][0]).toMatchObject({
      page_id: "next",
    });
    expect(callCloudProxy).not.toHaveBeenCalled();
    expect(outputs).toEqual([OUTPUT_1, OUTPUT_2]);
  });

  it("works without a conversation URL (falls back to backend host)", async () => {
    searchEventsMock.mockResolvedValueOnce({ items: [OUTPUT_1] });

    const outputs = await BashService.listOutputs(null, null, BASH_CMD_ID);

    expect(BashClient).toHaveBeenCalled();
    expect(callCloudProxy).not.toHaveBeenCalled();
    expect(outputs).toEqual([OUTPUT_1]);
  });
});

describe("BashService.listOutputs — cloud backend", () => {
  beforeEach(() => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id, orgId: null });
  });

  it("routes through callCloudProxy with hostOverride and session-api-key", async () => {
    vi.mocked(callCloudProxy).mockResolvedValueOnce({
      items: [OUTPUT_1, OUTPUT_2],
    });

    const outputs = await BashService.listOutputs(
      CONVERSATION_URL,
      SESSION_KEY,
      BASH_CMD_ID,
    );

    expect(BashClient).not.toHaveBeenCalled();
    const proxyCall = vi.mocked(callCloudProxy).mock.calls[0][0];
    expect(proxyCall.method).toBe("GET");
    expect(proxyCall.path).toMatch(/^\/api\/bash\/bash_events\/search\?/);
    expect(proxyCall.hostOverride).toBe("https://runtime.example.com");
    expect(proxyCall.authMode).toBe("session-api-key");
    expect(proxyCall.sessionApiKey).toBe(SESSION_KEY);

    const searchUrl = new URL(
      `http://x.example.com${proxyCall.path as string}`,
    );
    expect(searchUrl.searchParams.get("kind__eq")).toBe("BashOutput");
    expect(searchUrl.searchParams.get("command_id__eq")).toBe(BASH_CMD_ID);
    expect(searchUrl.searchParams.get("sort_order")).toBe("TIMESTAMP");

    expect(outputs).toEqual([OUTPUT_1, OUTPUT_2]);
  });

  it("throws when no conversation URL is provided on cloud backends", async () => {
    await expect(
      BashService.listOutputs(null, SESSION_KEY, BASH_CMD_ID),
    ).rejects.toThrow(/requires a conversation URL/);
    expect(callCloudProxy).not.toHaveBeenCalled();
  });
});
