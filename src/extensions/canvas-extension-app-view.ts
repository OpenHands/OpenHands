export interface CanvasExtensionAppViewSession {
  url: string;
  expiresAt: string;
  iframeSandbox: string;
}

export interface CanvasExtensionAppViewLabels {
  loading: string;
  retry: string;
  openInNewTab: string;
  unavailable: string;
}

export interface CanvasExtensionAppViewSessionContext {
  signal: AbortSignal;
}

export interface MountCanvasExtensionAppViewOptions {
  container: HTMLElement;
  labels: CanvasExtensionAppViewLabels;
  createSession: (
    context: CanvasExtensionAppViewSessionContext,
  ) => Promise<CanvasExtensionAppViewSession>;
  revokeSession?: () => Promise<void>;
}

export interface MountedCanvasExtensionAppView {
  dispose: () => void;
  retry: () => void;
}

const IFRAME_LOAD_TIMEOUT_MS = 15000;
const SAFE_SANDBOX_TOKENS = new Set([
  "allow-forms",
  "allow-modals",
  "allow-popups",
  "allow-same-origin",
  "allow-scripts",
]);

class InvalidCanvasExtensionAppViewSessionError extends Error {}

function validatedSession(
  session: CanvasExtensionAppViewSession,
): CanvasExtensionAppViewSession {
  let url: URL;
  try {
    url = new URL(session.url);
  } catch {
    throw new InvalidCanvasExtensionAppViewSessionError();
  }
  if (
    (url.protocol !== "http:" && url.protocol !== "https:") ||
    url.username ||
    url.password ||
    url.origin === window.location.origin
  ) {
    throw new InvalidCanvasExtensionAppViewSessionError();
  }

  const sandboxTokens = session.iframeSandbox.split(/\s+/).filter(Boolean);
  if (
    sandboxTokens.length === 0 ||
    sandboxTokens.some((token) => !SAFE_SANDBOX_TOKENS.has(token))
  ) {
    throw new InvalidCanvasExtensionAppViewSessionError();
  }

  return { ...session, url: url.href, iframeSandbox: sandboxTokens.join(" ") };
}

function createStatus(message: string): HTMLDivElement {
  const status = document.createElement("div");
  status.setAttribute("role", "status");
  status.className =
    "flex h-full min-h-64 w-full items-center justify-center text-sm text-muted";
  status.textContent = message;
  return status;
}

function createAction(label: string, onClick: () => void): HTMLButtonElement {
  const button = document.createElement("button");
  button.type = "button";
  button.className =
    "inline-flex min-h-10 cursor-pointer items-center justify-center rounded-lg bg-primary px-4 text-sm font-semibold text-[var(--oh-accent-foreground)] transition-opacity hover:opacity-80";
  button.textContent = label;
  button.addEventListener("click", onClick);
  return button;
}

export function mountCanvasExtensionAppView({
  container,
  labels,
  createSession,
  revokeSession,
}: MountCanvasExtensionAppViewOptions): MountedCanvasExtensionAppView {
  container.classList.add("h-full", "min-h-0", "w-full");
  let disposed = false;
  let generation = 0;
  let controller: AbortController | null = null;
  let loadTimeout: ReturnType<typeof setTimeout> | null = null;
  let activeMount: Promise<void> | null = null;
  let hasSession = false;

  const clearLoadTimeout = () => {
    if (loadTimeout === null) return;
    clearTimeout(loadTimeout);
    loadTimeout = null;
  };

  const revoke = async () => {
    if (!hasSession || !revokeSession) return;
    hasSession = false;
    try {
      await revokeSession();
    } catch (error) {
      console.error("Canvas App view session cleanup failed", error);
    }
  };

  const renderError = (message: string, url?: string) => {
    if (disposed) return;
    const wrapper = document.createElement("div");
    wrapper.className =
      "flex h-full min-h-64 w-full items-center justify-center p-8";
    const panel = document.createElement("div");
    panel.setAttribute("role", "alert");
    panel.className =
      "flex w-full max-w-lg flex-col items-center rounded-xl border border-border bg-surface-card p-8 text-center";
    const description = document.createElement("p");
    description.className =
      "mb-6 mt-0 max-w-md text-base font-medium leading-6 text-content";
    description.textContent = message;
    const actions = document.createElement("div");
    actions.className = "flex flex-wrap items-center justify-center gap-3";
    actions.append(createAction(labels.retry, retry));

    if (url) {
      const link = document.createElement("a");
      link.className =
        "inline-flex min-h-10 cursor-pointer items-center justify-center rounded-lg border border-border px-4 text-sm font-semibold text-content hover:bg-surface-raised";
      link.href = url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = labels.openInNewTab;
      link.addEventListener("click", (event) => {
        event.preventDefault();
        window.open(url, "_blank", "noopener,noreferrer");
      });
      actions.append(link);
    }
    panel.append(description, actions);
    wrapper.append(panel);
    container.replaceChildren(wrapper);
  };

  const mount = async () => {
    const currentGeneration = ++generation;
    controller?.abort();
    clearLoadTimeout();
    controller = new AbortController();
    container.replaceChildren(createStatus(labels.loading));

    try {
      const response = await createSession({
        signal: controller.signal,
      });
      hasSession = true;
      const created = validatedSession(response);
      if (disposed || currentGeneration !== generation) {
        await revoke();
        return;
      }

      const frame = document.createElement("iframe");
      frame.src = created.url;
      frame.title = labels.loading;
      frame.setAttribute("sandbox", created.iframeSandbox);
      frame.className = "h-full w-full border-0 bg-white";
      frame.addEventListener(
        "load",
        () => {
          if (!disposed && currentGeneration === generation) {
            clearLoadTimeout();
            container.replaceChildren(frame);
          }
        },
        { once: true },
      );
      frame.addEventListener(
        "error",
        () => {
          if (!disposed && currentGeneration === generation) {
            clearLoadTimeout();
            renderError(labels.unavailable, created.url);
          }
        },
        { once: true },
      );
      loadTimeout = setTimeout(() => {
        loadTimeout = null;
        if (!disposed && currentGeneration === generation) {
          renderError(labels.unavailable, created.url);
        }
      }, IFRAME_LOAD_TIMEOUT_MS);
      container.replaceChildren(createStatus(labels.loading), frame);
    } catch {
      if (hasSession) await revoke();
      if (disposed || currentGeneration !== generation) return;
      renderError(labels.unavailable);
    }
  };

  function retry() {
    if (disposed) return;
    generation += 1;
    controller?.abort();
    clearLoadTimeout();
    const previousMount = activeMount;
    activeMount = (async () => {
      await previousMount;
      await revoke();
      if (!disposed) await mount();
    })();
  }

  activeMount = mount();

  return {
    retry,
    dispose: () => {
      if (disposed) return;
      disposed = true;
      generation += 1;
      controller?.abort();
      clearLoadTimeout();
      container.replaceChildren();
      void revoke();
    },
  };
}
