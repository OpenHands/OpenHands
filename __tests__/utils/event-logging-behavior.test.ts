import { afterEach, describe, expect, it, vi } from "vitest";

const loadEventLogger = async (nodeEnv: string) => {
  vi.stubEnv("NODE_ENV", nodeEnv);
  vi.resetModules();

  return (await import("#/utils/event-logger")).default;
};

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
  vi.resetModules();
});

describe("event logging behavior", () => {
  it("pretty-prints parsed message data in development", async () => {
    const EventLogger = await loadEventLogger("development");
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const event = new MessageEvent("message", {
      data: '{"status":"ready","details":{"attempt":2}}',
    });

    EventLogger.message(event);

    expect(warn).toHaveBeenCalledOnce();
    expect(warn).toHaveBeenCalledWith(
      [
        "{",
        '  "status": "ready",',
        '  "details": {',
        '    "attempt": 2',
        "  }",
        "}",
      ].join("\n"),
    );
  });

  it("logs a named event with the original event object in development", async () => {
    const EventLogger = await loadEventLogger("development");
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const event = new Event("connection-opened");

    EventLogger.event(event, "CONNECTION_OPENED");

    expect(warn).toHaveBeenCalledOnce();
    expect(warn).toHaveBeenCalledWith("CONNECTION_OPENED", event);
  });

  it("uses the default name when a development event has no name", async () => {
    const EventLogger = await loadEventLogger("development");
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const event = new Event("unnamed");

    EventLogger.event(event);

    expect(warn).toHaveBeenCalledOnce();
    expect(warn).toHaveBeenCalledWith("EVENT", event);
  });

  it("routes development warnings and errors to their matching consoles", async () => {
    const EventLogger = await loadEventLogger("development");
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const error = vi.spyOn(console, "error").mockImplementation(() => {});

    EventLogger.warning("retrying connection");
    EventLogger.error("connection failed");

    expect(warn).toHaveBeenCalledOnce();
    expect(warn).toHaveBeenCalledWith("retrying connection");
    expect(error).toHaveBeenCalledOnce();
    expect(error).toHaveBeenCalledWith("connection failed");
  });

  it("suppresses every logging method outside development", async () => {
    const EventLogger = await loadEventLogger("production");
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const error = vi.spyOn(console, "error").mockImplementation(() => {});

    EventLogger.message(
      new MessageEvent("message", { data: '{"status":"ignored"}' }),
    );
    EventLogger.event(new Event("ignored"), "IGNORED");
    EventLogger.warning("ignored warning");
    EventLogger.error("ignored error");

    expect(warn).not.toHaveBeenCalled();
    expect(error).not.toHaveBeenCalled();
  });
});
