import { afterEach, describe, expect, it } from "vitest";
import {
  AGENT_SERVER_UNKNOWN_VERSION_ERROR_CODE,
  AGENT_SERVER_UNSUPPORTED_VERSION_ERROR_CODE,
  AgentServerUnavailableError,
  AgentServerUnknownVersionError,
  AgentServerUnsupportedVersionError,
  assertAgentServerVersionIsSupported,
  getDisplayAgentServerVersion,
  isAgentServerAuthError,
  isAgentServerUnavailableError,
  isAgentServerUnknownVersionError,
  isAgentServerUnsupportedVersionError,
  isSdkHttpError,
  isSdkHttpStatusError,
  MINIMUM_COMPATIBLE_AGENT_SERVER_VERSION,
  type AgentServerInfo,
} from "./agent-server-compatibility";

const serverInfo = (version?: unknown): AgentServerInfo =>
  ({ version }) as AgentServerInfo;

const httpError = (status: unknown): Error & { status: unknown } =>
  Object.assign(new Error(`HTTP ${String(status)}`), {
    name: "HttpError",
    status,
  });

const getThrownError = (action: () => void): Error => {
  try {
    action();
  } catch (error) {
    if (error instanceof Error) return error;
    throw error;
  }
  throw new Error("Expected action to throw");
};

afterEach(() => {
  delete (window as unknown as Record<string, unknown>)
    .__AGENT_CANVAS_AUTH_REQUIRED__;
});

describe("agent-server version compatibility", () => {
  it.each([
    { version: undefined, reported: null },
    { version: 128, reported: null },
    { version: "", reported: null },
    { version: "   ", reported: null },
    { version: "unknown", reported: "unknown" },
    { version: " UNKNOWN ", reported: "UNKNOWN" },
  ])("classifies $version as an unknown version", ({ version, reported }) => {
    const error = getThrownError(() =>
      assertAgentServerVersionIsSupported(serverInfo(version)),
    );

    expect(error).toBeInstanceOf(AgentServerUnknownVersionError);
    expect(error).toMatchObject({
      code: AGENT_SERVER_UNKNOWN_VERSION_ERROR_CODE,
      actualVersion: reported,
      requiredVersion: MINIMUM_COMPATIBLE_AGENT_SERVER_VERSION,
    });
  });

  it.each([
    "dev-build",
    "1.28",
    "1.28.0.1",
    "1.two.0",
    "1.28.v0",
    "Infinity.28.0",
    "V1.28.0",
  ])("classifies malformed version %s as unknown", (version) => {
    const error = getThrownError(() =>
      assertAgentServerVersionIsSupported(serverInfo(version)),
    );

    expect(error).toBeInstanceOf(AgentServerUnknownVersionError);
    expect(error).toMatchObject({
      actualVersion: version,
      code: AGENT_SERVER_UNKNOWN_VERSION_ERROR_CODE,
    });
  });

  it.each(["0.99.99", "1.27.999", "1.28.0-rc.1"])(
    "rejects older version %s",
    (version) => {
      const error = getThrownError(() =>
        assertAgentServerVersionIsSupported(serverInfo(version)),
      );

      expect(error).toBeInstanceOf(AgentServerUnsupportedVersionError);
      expect(error).toMatchObject({
        actualVersion: version,
        code: AGENT_SERVER_UNSUPPORTED_VERSION_ERROR_CODE,
        requiredVersion: MINIMUM_COMPATIBLE_AGENT_SERVER_VERSION,
      });
    },
  );

  it.each([
    MINIMUM_COMPATIBLE_AGENT_SERVER_VERSION,
    "1.28.1",
    "1.29.0",
    "2.0.0",
    " v1.28.0+build.7 ",
  ])("accepts compatible version %s", (version) => {
    expect(() =>
      assertAgentServerVersionIsSupported(serverInfo(version)),
    ).not.toThrow();
  });

  it.each([
    { version: undefined, display: null },
    { version: "unknown", display: null },
    { version: "dev-build", display: null },
    { version: "1.28.0", display: "1.28.0" },
    { version: " v1.28.0+build.7 ", display: "v1.28.0+build.7" },
    { version: "1.28.0-rc.1", display: "1.28.0-rc.1" },
  ])("displays $version as $display", ({ version, display }) => {
    expect(getDisplayAgentServerVersion(serverInfo(version))).toBe(display);
  });
});

describe("agent-server compatibility errors", () => {
  it("publishes stable error codes for callers", () => {
    expect(AGENT_SERVER_UNSUPPORTED_VERSION_ERROR_CODE).toBe(
      "AGENT_SERVER_UNSUPPORTED_VERSION",
    );
    expect(AGENT_SERVER_UNKNOWN_VERSION_ERROR_CODE).toBe(
      "AGENT_SERVER_UNKNOWN_VERSION",
    );
  });

  it("constructs unavailable errors for connection and missing-backend failures", () => {
    expect(new AgentServerUnavailableError()).toMatchObject({
      name: "AgentServerUnavailableError",
      message:
        "Could not connect to the configured agent server. Make sure it is running and reachable, then reload the page.",
      details: null,
      noBackendConfigured: false,
    });
    expect(
      new AgentServerUnavailableError("No backend configured", {
        noBackendConfigured: true,
      }),
    ).toMatchObject({
      name: "AgentServerUnavailableError",
      message:
        "No agent server backend is configured yet. Add a backend to get started.",
      details: "No backend configured",
      noBackendConfigured: true,
    });
  });

  it("describes unsupported and unknown versions exactly", () => {
    expect(new AgentServerUnsupportedVersionError("1.27.0").message).toBe(
      `Agent Canvas requires agent-server ${MINIMUM_COMPATIBLE_AGENT_SERVER_VERSION} or newer; this backend is running 1.27.0. Please upgrade the agent-server backend.`,
    );
    expect(new AgentServerUnknownVersionError(null).message).toBe(
      `Could not determine this backend's agent-server version. Agent Canvas requires agent-server ${MINIMUM_COMPATIBLE_AGENT_SERVER_VERSION} or newer, but this backend did not return a valid version from /server_info. Restart or rebuild the agent-server backend, then try again.`,
    );
    expect(new AgentServerUnknownVersionError("dev-build").message).toBe(
      `Could not determine this backend's agent-server version. It reported "dev-build". Agent Canvas requires agent-server ${MINIMUM_COMPATIBLE_AGENT_SERVER_VERSION} or newer, but this backend did not return a valid version from /server_info. Restart or rebuild the agent-server backend, then try again.`,
    );
  });

  it.each([
    new AgentServerUnavailableError(),
    { name: "AgentServerUnavailableError" },
  ])("recognizes unavailable error %#", (error) => {
    expect(isAgentServerUnavailableError(error)).toBe(true);
  });

  it.each([null, "AgentServerUnavailableError", {}, { name: "AnotherError" }])(
    "rejects unavailable lookalike %#",
    (error) => {
      expect(isAgentServerUnavailableError(error)).toBe(false);
    },
  );

  it.each([
    new AgentServerUnsupportedVersionError("1.27.0"),
    { code: AGENT_SERVER_UNSUPPORTED_VERSION_ERROR_CODE },
  ])("recognizes unsupported-version error %#", (error) => {
    expect(isAgentServerUnsupportedVersionError(error)).toBe(true);
  });

  it.each([
    null,
    "AGENT_SERVER_UNSUPPORTED_VERSION",
    {},
    { code: "ANOTHER_ERROR" },
  ])("rejects unsupported-version lookalike %#", (error) => {
    expect(isAgentServerUnsupportedVersionError(error)).toBe(false);
  });

  it.each([
    new AgentServerUnknownVersionError("dev-build"),
    { code: AGENT_SERVER_UNKNOWN_VERSION_ERROR_CODE },
  ])("recognizes unknown-version error %#", (error) => {
    expect(isAgentServerUnknownVersionError(error)).toBe(true);
  });

  it.each([
    null,
    "AGENT_SERVER_UNKNOWN_VERSION",
    {},
    { code: "ANOTHER_ERROR" },
  ])("rejects unknown-version lookalike %#", (error) => {
    expect(isAgentServerUnknownVersionError(error)).toBe(false);
  });
});

describe("SDK HTTP error classification", () => {
  it("requires an Error named HttpError with a numeric status", () => {
    expect(isSdkHttpError(httpError(401))).toBe(true);
    expect(isSdkHttpError({ name: "HttpError", status: 401 })).toBe(false);
    expect(isSdkHttpError(new Error("HTTP 401"))).toBe(false);
    expect(
      isSdkHttpError(
        Object.assign(new Error("HTTP 401"), {
          name: "AnotherError",
          status: 401,
        }),
      ),
    ).toBe(false);
    expect(
      isSdkHttpError(Object.assign(new Error(), { name: "HttpError" })),
    ).toBe(false);
    expect(isSdkHttpError(httpError("401"))).toBe(false);
  });

  it("matches only the requested HTTP status", () => {
    expect(isSdkHttpStatusError(httpError(401), 401)).toBe(true);
    expect(isSdkHttpStatusError(httpError(403), 401)).toBe(false);
    expect(isSdkHttpStatusError(new Error("network"), 401)).toBe(false);
  });

  it("only treats 401 as an auth error when public auth is required", () => {
    const authWindow = window as unknown as Record<string, unknown>;
    expect(isAgentServerAuthError(httpError(401))).toBe(false);

    authWindow.__AGENT_CANVAS_AUTH_REQUIRED__ = true;
    expect(isAgentServerAuthError(httpError(401))).toBe(true);
    expect(isAgentServerAuthError(httpError(403))).toBe(false);
  });
});
