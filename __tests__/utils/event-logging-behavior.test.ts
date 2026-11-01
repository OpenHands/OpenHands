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
  it("routes development errors to the error console", async () => {
    const EventLogger = await loadEventLogger("development");
    const error = vi.spyOn(console, "error").mockImplementation(() => {});

    EventLogger.error("connection failed");

    expect(error).toHaveBeenCalledOnce();
    expect(error).toHaveBeenCalledWith("connection failed");
  });

  it("suppresses logging outside development", async () => {
    const EventLogger = await loadEventLogger("production");
    const error = vi.spyOn(console, "error").mockImplementation(() => {});

    EventLogger.error("ignored error");

    expect(error).not.toHaveBeenCalled();
  });
});
