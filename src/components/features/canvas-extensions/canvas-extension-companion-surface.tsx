import React from "react";
import {
  type RegisteredCanvasExtensionCompanion,
  useCanvasExtensionsRuntime,
} from "./canvas-extensions-runtime";

function Companion({
  companion,
}: {
  companion: RegisteredCanvasExtensionCompanion;
}) {
  const containerRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    const container = containerRef.current;
    if (!container) return undefined;
    let disposed = false;
    let disposeMount: (() => void) | undefined;
    const safelyDispose = (dispose?: () => void) => {
      try {
        dispose?.();
      } catch (error) {
        console.error("App companion cleanup failed", error);
      }
    };
    Promise.resolve()
      .then(() => {
        if (disposed) return undefined;
        return companion.mount({ container, navigate: companion.navigate });
      })
      .then((dispose) => {
        if (typeof dispose !== "function") return;
        if (disposed) safelyDispose(dispose);
        else disposeMount = dispose;
      })
      .catch((error: unknown) => {
        if (!disposed) {
          container.replaceChildren();
          console.error("App companion failed", error);
        }
      });
    return () => {
      disposed = true;
      safelyDispose(disposeMount);
      container.replaceChildren();
    };
  }, [companion]);

  return (
    <div
      ref={containerRef}
      data-app-companion={`${companion.extensionName}/${companion.id}`}
      className="min-w-0 max-w-full shrink-0"
    />
  );
}

/** A persistent shell dock: it shares space with the view rather than covering it. */
export function CanvasExtensionCompanionDock() {
  const { companions } = useCanvasExtensionsRuntime();
  if (companions.length === 0) return null;

  return (
    <div
      data-testid="canvas-extension-companion-dock"
      className="flex min-h-0 w-full max-h-[min(40dvh,20rem)] shrink-0 flex-col items-end overflow-y-auto custom-scrollbar"
    >
      {companions.map((companion) => (
        <Companion
          key={`${companion.scope}/${companion.extensionName}/${companion.id}`}
          companion={companion}
        />
      ))}
    </div>
  );
}
