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

function createStatus(message: string): HTMLParagraphElement {
  const status = document.createElement("p");
  status.setAttribute("role", "status");
  status.textContent = message;
  return status;
}

function createAction(label: string, onClick: () => void): HTMLButtonElement {
  const button = document.createElement("button");
  button.type = "button";
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
  let disposed = false;
  let generation = 0;
  let controller: AbortController | null = null;
  let hasSession = false;

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
    const panel = document.createElement("div");
    panel.setAttribute("role", "alert");
    const description = document.createElement("p");
    description.textContent = message;
    panel.append(description, createAction(labels.retry, retry));

    if (url) {
      const link = document.createElement("a");
      link.href = url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = labels.openInNewTab;
      link.addEventListener("click", (event) => {
        event.preventDefault();
        window.open(url, "_blank", "noopener,noreferrer");
      });
      panel.append(link);
    }
    container.replaceChildren(panel);
  };

  const mount = async () => {
    const currentGeneration = ++generation;
    controller?.abort();
    controller = new AbortController();
    container.replaceChildren(createStatus(labels.loading));

    try {
      const response = await createSession({ signal: controller.signal });
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
            container.replaceChildren(frame);
          }
        },
        { once: true },
      );
      frame.addEventListener(
        "error",
        () => {
          if (!disposed && currentGeneration === generation) {
            renderError(labels.unavailable, created.url);
          }
        },
        { once: true },
      );
      container.replaceChildren(createStatus(labels.loading), frame);
    } catch {
      if (hasSession) await revoke();
      if (disposed || currentGeneration !== generation) return;
      renderError(labels.unavailable);
    }
  };

  function retry() {
    if (disposed) return;
    controller?.abort();
    void revoke().finally(mount);
  }

  void mount();

  return {
    retry,
    dispose: () => {
      if (disposed) return;
      disposed = true;
      generation += 1;
      controller?.abort();
      container.replaceChildren();
      void revoke();
    },
  };
}
