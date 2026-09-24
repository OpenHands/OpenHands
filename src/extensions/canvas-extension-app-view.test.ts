import { fireEvent } from "@testing-library/dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  mountCanvasExtensionAppView,
  type CanvasExtensionAppViewLabels,
  type CanvasExtensionAppViewSession,
} from "./canvas-extension-app-view";

const labels: CanvasExtensionAppViewLabels = {
  loading: "Loading app view",
  retry: "Retry",
  openInNewTab: "Open in new tab",
  unavailable: "App view unavailable",
};

const session: CanvasExtensionAppViewSession = {
  url: "https://apps.example.test/app-backends/demo/",
  expiresAt: "2026-08-14T06:10:00Z",
  iframeSandbox:
    "allow-forms allow-modals allow-popups allow-same-origin allow-scripts",
};

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

afterEach(() => {
  vi.restoreAllMocks();
  document.body.replaceChildren();
});

describe("mountCanvasExtensionAppView", () => {
  it("mounts a server-provided HTTPS view in an isolated frame", async () => {
    const container = document.createElement("div");
    document.body.append(container);

    mountCanvasExtensionAppView({
      container,
      labels,
      createSession: vi.fn().mockResolvedValue(session),
    });

    expect(container).toHaveTextContent(labels.loading);
    const frame = await vi.waitUntil(() => container.querySelector("iframe"));
    expect(frame).toHaveAttribute("src", session.url);
    expect(frame).toHaveAttribute("sandbox", session.iframeSandbox);
    expect(frame).toHaveAttribute(
      "sandbox",
      expect.stringContaining("allow-same-origin"),
    );

    fireEvent.load(frame);
    expect(container).not.toHaveTextContent(labels.loading);
  });

  it.each([
    "not a URL",
    "javascript:alert(1)",
    "data:text/html,unsafe",
    "file:///tmp/app",
    `${window.location.origin}/app-backends/demo/`,
  ])(
    "rejects an unsafe bootstrap URL and revokes its session: %s",
    async (url) => {
      const container = document.createElement("div");
      const revokeSession = vi.fn().mockResolvedValue(undefined);

      mountCanvasExtensionAppView({
        container,
        labels,
        createSession: vi.fn().mockResolvedValue({ ...session, url }),
        revokeSession,
      });

      await vi.waitFor(() =>
        expect(container).toHaveTextContent(labels.unavailable),
      );
      expect(revokeSession).toHaveBeenCalledTimes(1);
      expect(container.querySelector("iframe")).toBeNull();
      expect(container.querySelector("a")).toBeNull();
    },
  );

  it("rejects sandbox permissions outside the bridge isolation contract", async () => {
    const container = document.createElement("div");

    mountCanvasExtensionAppView({
      container,
      labels,
      createSession: vi.fn().mockResolvedValue({
        ...session,
        iframeSandbox: `${session.iframeSandbox} allow-top-navigation`,
      }),
    });

    await vi.waitFor(() =>
      expect(container).toHaveTextContent(labels.unavailable),
    );
    expect(container.querySelector("iframe")).toBeNull();
  });

  it("shows an error with retry and opens only the validated URL", async () => {
    const container = document.createElement("div");
    const createSession = vi
      .fn<() => Promise<CanvasExtensionAppViewSession>>()
      .mockRejectedValueOnce(new Error("backend not ready"))
      .mockResolvedValueOnce(session);
    const open = vi.spyOn(window, "open").mockImplementation(() => null);

    mountCanvasExtensionAppView({ container, labels, createSession });

    await vi.waitFor(() =>
      expect(container).toHaveTextContent(labels.unavailable),
    );
    expect(container).not.toHaveTextContent("backend not ready");
    fireEvent.click(container.querySelector("button")!);
    const frame = await vi.waitUntil(() => container.querySelector("iframe"));
    fireEvent.error(frame);

    const link = await vi.waitUntil(() => container.querySelector("a"));
    fireEvent.click(link);
    expect(open).toHaveBeenCalledWith(
      session.url,
      "_blank",
      "noopener,noreferrer",
    );
    expect(createSession).toHaveBeenCalledTimes(2);
  });

  it("aborts a pending mount and revokes a late session on disposal", async () => {
    const container = document.createElement("div");
    const pending = deferred<CanvasExtensionAppViewSession>();
    const revokeSession = vi.fn().mockResolvedValue(undefined);
    let signal: AbortSignal | undefined;
    const mounted = mountCanvasExtensionAppView({
      container,
      labels,
      createSession: ({ signal: nextSignal }) => {
        signal = nextSignal;
        return pending.promise;
      },
      revokeSession,
    });

    mounted.dispose();
    expect(signal?.aborted).toBe(true);
    expect(container).toBeEmptyDOMElement();

    pending.resolve(session);
    await vi.waitFor(() => expect(revokeSession).toHaveBeenCalledTimes(1));
    expect(container).toBeEmptyDOMElement();
  });

  it("revokes the current session before replacing or disposing its view", async () => {
    const container = document.createElement("div");
    const revokeSession = vi.fn().mockResolvedValue(undefined);
    const mounted = mountCanvasExtensionAppView({
      container,
      labels,
      createSession: vi.fn().mockResolvedValue(session),
      revokeSession,
    });

    const firstFrame = await vi.waitUntil(() =>
      container.querySelector("iframe"),
    );
    fireEvent.error(firstFrame);
    fireEvent.click(container.querySelector("button")!);

    await vi.waitFor(() => expect(revokeSession).toHaveBeenCalledTimes(1));
    await vi.waitUntil(() => container.querySelector("iframe") !== firstFrame);
    mounted.dispose();
    await vi.waitFor(() => expect(revokeSession).toHaveBeenCalledTimes(2));
    expect(container).toBeEmptyDOMElement();
  });

  it("does not remount when disposed while retry is revoking", async () => {
    const container = document.createElement("div");
    const revocation = deferred<void>();
    const createSession = vi.fn().mockResolvedValue(session);
    const mounted = mountCanvasExtensionAppView({
      container,
      labels,
      createSession,
      revokeSession: () => revocation.promise,
    });

    const frame = await vi.waitUntil(() => container.querySelector("iframe"));
    fireEvent.error(frame);
    fireEvent.click(container.querySelector("button")!);
    mounted.dispose();
    revocation.resolve();

    await Promise.resolve();
    expect(createSession).toHaveBeenCalledTimes(1);
    expect(container).toBeEmptyDOMElement();
  });
});
