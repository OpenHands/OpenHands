import React from "react";
import { useNavigate, useParams } from "react-router";
import { useTranslation } from "react-i18next";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { useCanvasExtensionsRuntime } from "#/components/features/canvas-extensions/canvas-extensions-runtime";
import { I18nKey } from "#/i18n/declaration";

export default function CanvasExtensionPage() {
  const { t } = useTranslation("openhands");
  const params = useParams();
  const navigate = useNavigate();
  const extensionName = params.extensionName ?? "";
  const routePath = params["*"] ?? "";
  const { pages, activating, errors } = useCanvasExtensionsRuntime();
  const containerRef = React.useRef<HTMLDivElement>(null);
  const [mountError, setMountError] = React.useState<string | null>(null);

  const page = React.useMemo(
    () =>
      pages
        .filter((candidate) => candidate.extension.name === extensionName)
        .sort(
          (left, right) =>
            right.contribution.path.length - left.contribution.path.length,
        )
        .find(
          (candidate) =>
            routePath === candidate.contribution.path ||
            routePath.startsWith(`${candidate.contribution.path}/`),
        ),
    [extensionName, pages, routePath],
  );

  React.useEffect(() => {
    const container = containerRef.current;
    if (!container || !page) return undefined;
    let disposed = false;
    let disposeMount: (() => void) | undefined;
    setMountError(null);
    container.replaceChildren();

    const remainder = routePath
      .slice(page.contribution.path.length)
      .replace(/^\/+/, "");
    Promise.resolve()
      .then(() =>
        page.mount({
          container,
          path: remainder,
          navigate: (path) => navigate(path),
        }),
      )
      .then((dispose) => {
        if (typeof dispose !== "function") return;
        if (disposed) dispose();
        else disposeMount = dispose;
      })
      .catch((error: unknown) => {
        if (!disposed) {
          setMountError(
            error instanceof Error ? error.message : "Extension page failed.",
          );
        }
      });

    return () => {
      disposed = true;
      try {
        disposeMount?.();
      } catch (error) {
        console.error("Canvas Extension page cleanup failed", error);
      }
      container.replaceChildren();
    };
  }, [navigate, page, routePath]);

  if (activating && !page) {
    return (
      <div className="flex h-full items-center justify-center">
        <LoadingSpinner size="large" />
      </div>
    );
  }

  const error = mountError ?? errors.get(extensionName);
  if (error || !page) {
    return (
      <main className="flex h-full items-center justify-center p-8">
        <div className="max-w-lg rounded-xl border border-[var(--oh-border)] bg-base-secondary p-6 text-center">
          <h1 className="text-lg font-semibold text-white">
            {t(I18nKey.SETUP$UNAVAILABLE_TITLE)}
          </h1>
          <p className="mt-2 text-sm text-tertiary-light">
            {error || t(I18nKey.SETTINGS$CANVAS_EXTENSIONS_PAGE_UNAVAILABLE)}
          </p>
        </div>
      </main>
    );
  }

  return (
    <main
      aria-label={page.contribution.title}
      className="h-full min-h-0 overflow-auto"
    >
      <div ref={containerRef} className="min-h-full" />
    </main>
  );
}
