import { describe, expect, it } from "vitest";
import i18next, { type TFunction } from "i18next";
import { translationResources } from "#/i18n/resources";
import { makeMcpTestErrorMessage } from "#/utils/mcp-test-error-message";

// Drive a real i18next instance with the actual en resource so the assertions
// cover the rendered copy and its interpolation, not just the chosen key.
const instance = i18next.createInstance();
instance.init({
  lng: "en",
  resources: { en: { openhands: translationResources.en } },
  ns: ["openhands"],
  defaultNS: "openhands",
  interpolation: { escapeValue: false },
});
const t = instance.t as unknown as TFunction<"openhands">;

const STDIO_DETAIL =
  "Client failed to connect: [Errno 2] No such file or directory: 'qa-no-such-command'";
const REMOTE_DETAIL =
  "Client failed to connect: All connection attempts failed";

describe("makeMcpTestErrorMessage", () => {
  it("explains a stdio connection failure as a command that could not start, with the backend detail", () => {
    const message = makeMcpTestErrorMessage(
      t,
      "connection",
      STDIO_DETAIL,
      "stdio",
    );

    expect(message).toBe(`Could not start the server command: ${STDIO_DETAIL}`);
    expect(message).not.toContain("URL");
  });

  it.each(["shttp", "sse", undefined] as const)(
    "keeps the reach-the-server guidance for a %s connection failure and adds the backend detail",
    (serverType) => {
      const message = makeMcpTestErrorMessage(
        t,
        "connection",
        REMOTE_DETAIL,
        serverType,
      );

      expect(message).toBe(
        `Could not reach the server (check the URL and server type): ${REMOTE_DETAIL}`,
      );
    },
  );

  it("tells the user to finish the sign-in when an OAuth authorization was not completed", () => {
    const message = makeMcpTestErrorMessage(
      t,
      "authorization",
      "OAuth authorization was not completed",
      "shttp",
    );

    expect(message).toBe(
      "Sign-in was not completed. Finish signing in from the pop-up window, then try again.",
    );
    expect(message).not.toContain("URL");
  });

  it("leaves timeout and credentials failures unchanged for stdio servers", () => {
    expect(makeMcpTestErrorMessage(t, "timeout", "late", "stdio")).toBe(
      "Connection timed out. Check the URL and try again.",
    );
    expect(
      makeMcpTestErrorMessage(t, "credentials", "invalid_auth", "stdio"),
    ).toBe("Credential check failed: invalid_auth");
  });

  it.each([
    "<!DOCTYPE html> <html><body>Not found</body></html>",
    '<html lang="en"><body>Not found</body></html>',
    "<!doctype HTML>\n<HTML>Not found</HTML>",
  ])(
    "omits an HTML response body while retaining the failure detail: %s",
    (body) => {
      const message = makeMcpTestErrorMessage(
        t,
        "connection",
        `Client failed to connect: Registration failed: 404 \n${body}`,
        "shttp",
      );

      expect(message).toBe(
        "Could not reach the server (check the URL and server type): Client failed to connect: Registration failed: 404",
      );
    },
  );

  it.each([
    ["connection", "stdio", "Could not start the server command"],
    ["credentials", "shttp", "Credential check failed"],
    ["unknown", "sse", "Connection failed"],
  ] as const)(
    "omits HTML for a %s error from a %s server",
    (kind, serverType, prefix) => {
      expect(
        makeMcpTestErrorMessage(
          t,
          kind,
          "Request failed: 502 <html><body>Bad gateway</body></html>",
          serverType,
        ),
      ).toBe(`${prefix}: Request failed: 502`);
    },
  );

  it.each([
    "  invalid_auth\nTry again.  ",
    "Expected <token>, got <htmlish> instead",
  ])("preserves non-HTML error details verbatim: %s", (detail) => {
    expect(makeMcpTestErrorMessage(t, "credentials", detail, "shttp")).toBe(
      `Credential check failed: ${detail}`,
    );
  });
});
