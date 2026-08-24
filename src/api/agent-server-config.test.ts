import { afterEach, describe, expect, it, vi } from "vitest";
import {
  DEFAULT_WORKING_DIR,
  buildConversationWorkingDir,
  buildConversationWorkingDirForBackend,
  buildRelativeConversationWorkingDir,
  isServedOriginHost,
} from "./agent-server-config";

describe("buildRelativeConversationWorkingDir", () => {
  afterEach(() => vi.unstubAllEnvs());

  it("ignores a baked absolute VITE_WORKING_DIR and stays relative", () => {
    vi.stubEnv("VITE_WORKING_DIR", "/Users/someone/.openhands/workspaces");
    expect(buildRelativeConversationWorkingDir("abc-123")).toBe(
      `${DEFAULT_WORKING_DIR}/abc123`,
    );
  });

  it("strips dashes from the conversation id", () => {
    expect(buildRelativeConversationWorkingDir("a-b-c")).toBe(
      `${DEFAULT_WORKING_DIR}/abc`,
    );
  });
});

describe("buildConversationWorkingDir (baked default)", () => {
  afterEach(() => vi.unstubAllEnvs());

  it("uses a baked absolute dir as the base", () => {
    vi.stubEnv("VITE_WORKING_DIR", "/Users/someone/workspaces");
    expect(buildConversationWorkingDir("abc-123")).toBe(
      "/Users/someone/workspaces/abc123",
    );
  });

  it("falls back to the relative default when nothing is baked", () => {
    vi.stubEnv("VITE_WORKING_DIR", "");
    expect(buildConversationWorkingDir("abc-123")).toBe(
      `${DEFAULT_WORKING_DIR}/abc123`,
    );
  });
});

describe("isServedOriginHost", () => {
  afterEach(() => vi.unstubAllEnvs());

  it("matches the host that served this frontend", () => {
    vi.stubEnv("VITE_BACKEND_BASE_URL", "http://127.0.0.1:8000");
    expect(isServedOriginHost("http://127.0.0.1:8000")).toBe(true);
  });

  it("normalizes a bare host against the served origin", () => {
    vi.stubEnv("VITE_BACKEND_BASE_URL", "http://127.0.0.1:8000");
    // getAgentServerBaseUrl returns the already-normalized configured URL;
    // a bare candidate host is normalized to http:// off-window.
    expect(isServedOriginHost("127.0.0.1:8000")).toBe(true);
  });

  it("does not match a different host", () => {
    vi.stubEnv("VITE_BACKEND_BASE_URL", "http://127.0.0.1:8000");
    expect(isServedOriginHost("https://cabin.liberty-labs.org")).toBe(false);
  });

  it("does not match when the backend host is empty", () => {
    vi.stubEnv("VITE_BACKEND_BASE_URL", "http://127.0.0.1:8000");
    expect(isServedOriginHost("")).toBe(false);
    expect(isServedOriginHost(null)).toBe(false);
  });
});

describe("buildConversationWorkingDirForBackend", () => {
  afterEach(() => vi.unstubAllEnvs());

  it("uses the baked absolute default for the served-origin backend", () => {
    vi.stubEnv("VITE_BACKEND_BASE_URL", "http://127.0.0.1:8000");
    vi.stubEnv(
      "VITE_WORKING_DIR",
      "/Users/me/.openhands/agent-canvas/workspaces",
    );
    expect(
      buildConversationWorkingDirForBackend("abc-123", "http://127.0.0.1:8000"),
    ).toBe("/Users/me/.openhands/agent-canvas/workspaces/abc123");
  });

  // Regression for the reviewed bug: the seeded `default-local` backend is
  // mutable. Editing its host to a remote VM (while its id stays
  // `default-local`) must NOT send the baked macOS path to that VM — this is
  // the exact case that failed on our remote Linux backend (Cabin) with
  // `mkdir: cannot create directory '/Users': Permission denied`.
  it("uses the relative default for an edited default-local pointing at a remote host", () => {
    vi.stubEnv("VITE_BACKEND_BASE_URL", "http://127.0.0.1:8000");
    vi.stubEnv(
      "VITE_WORKING_DIR",
      "/Users/me/.openhands/agent-canvas/workspaces",
    );
    // The user edited the seeded `default-local` entry to a remote Linux VM
    // (home /home/daytona); its host no longer matches the served origin.
    expect(
      buildConversationWorkingDirForBackend(
        "abc-123",
        "https://cabin.liberty-labs.org",
      ),
    ).toBe(`${DEFAULT_WORKING_DIR}/abc123`);
  });

  it("uses the relative default for any separately-registered remote backend", () => {
    vi.stubEnv("VITE_BACKEND_BASE_URL", "http://127.0.0.1:8000");
    vi.stubEnv("VITE_WORKING_DIR", "/Users/me/workspaces");
    expect(
      buildConversationWorkingDirForBackend(
        "abc-123",
        "https://sandbox.example.dev",
      ),
    ).toBe(`${DEFAULT_WORKING_DIR}/abc123`);
  });
});
