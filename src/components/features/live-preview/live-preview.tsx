import { useEffect, useMemo, useState } from "react";
import {
  ExternalLink,
  Loader2,
  RefreshCw,
  Rocket,
  WifiOff,
} from "lucide-react";
import { useTranslation } from "react-i18next";

import { I18nKey } from "#/i18n/declaration";

import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import {
  joinWorkspaceUrl,
  useWorkspaceSession,
} from "#/hooks/query/use-workspace-session";
import { useWorkspaceFiles } from "#/hooks/query/use-workspace-files";
import {
  useWorkspaceMutationCounter,
  withWorkspaceCacheBuster,
} from "#/stores/use-workspace-mutation-counter";
import { useLivePreviewStore } from "#/stores/live-preview-store";
import { useRuntimeIsReady } from "#/hooks/use-runtime-is-ready";
import { useAutoRefreshPreviewOnEdit } from "#/hooks/use-auto-refresh-preview-on-edit";
import { cn } from "#/utils/utils";

const ENTRYPOINT_PRIORITY = [
  "index.html",
  "public/index.html",
  "src/index.html",
  "app/index.html",
];

function normalizeWorkspacePath(path: string): string {
  return path.replace(/^\.\//, "").replace(/^\//, "");
}

function chooseEntrypoint(files: string[] | undefined, requestedPath?: string) {
  const normalizedFiles = (files ?? []).map(normalizeWorkspacePath);
  const requested = requestedPath
    ? normalizeWorkspacePath(requestedPath)
    : null;
  if (requested && normalizedFiles.includes(requested)) return requested;
  return (
    ENTRYPOINT_PRIORITY.find((path) => normalizedFiles.includes(path)) ?? null
  );
}

export function LivePreview() {
  const { t } = useTranslation("openhands");
  const { data: conversation } = useActiveConversation();
  const runtimeIsReady = useRuntimeIsReady();
  const workspaceSession = useWorkspaceSession();
  const workspaceFiles = useWorkspaceFiles();
  const mutationCounter = useWorkspaceMutationCounter((state) => state.count);
  const bumpMutationCounter = useWorkspaceMutationCounter(
    (state) => state.bump,
  );
  useAutoRefreshPreviewOnEdit();
  const requestedPath = useLivePreviewStore(
    (state) => state.requestedPaths[conversation?.id ?? ""],
  );
  const [loaded, setLoaded] = useState(false);
  const [loadError, setLoadError] = useState(false);
  const [manualRefresh, setManualRefresh] = useState(0);

  const entrypoint = useMemo(
    () => chooseEntrypoint(workspaceFiles.data, requestedPath),
    [workspaceFiles.data, requestedPath],
  );

  const previewUrl = useMemo(() => {
    if (!workspaceSession.data?.baseUrl || !entrypoint) return null;
    return withWorkspaceCacheBuster(
      joinWorkspaceUrl(workspaceSession.data.baseUrl, entrypoint),
      mutationCounter + manualRefresh,
    );
  }, [
    workspaceSession.data?.baseUrl,
    entrypoint,
    mutationCounter,
    manualRefresh,
  ]);

  useEffect(() => {
    setLoaded(false);
    setLoadError(false);
  }, [previewUrl]);

  const openPreview = () => {
    if (previewUrl) window.open(previewUrl, "_blank", "noopener,noreferrer");
  };

  const refreshPreview = () => {
    setLoaded(false);
    setLoadError(false);
    setManualRefresh((value) => value + 1);
    bumpMutationCounter();
  };

  if (!runtimeIsReady) {
    return (
      <div className="flex h-full min-h-0 w-full flex-col items-center justify-center gap-3 bg-[var(--oh-surface)] p-8 text-center text-[var(--oh-muted)]">
        <Loader2 className="h-6 w-6 animate-spin" aria-hidden />
        <p className="text-sm">
          {t(I18nKey.CONVERSATION_PANEL$PREVIEW_WAITING_RUNTIME)}
        </p>
      </div>
    );
  }

  if (workspaceSession.isError || workspaceFiles.isLoading) {
    return (
      <div className="flex h-full min-h-0 w-full flex-col items-center justify-center gap-3 bg-[var(--oh-surface)] p-8 text-center text-[var(--oh-muted)]">
        <WifiOff className="h-6 w-6" aria-hidden />
        <p className="text-sm">
          {t(I18nKey.CONVERSATION_PANEL$PREVIEW_UNAVAILABLE)}
        </p>
      </div>
    );
  }

  if (!entrypoint || !previewUrl) {
    return (
      <div className="flex h-full min-h-0 w-full flex-col items-center justify-center gap-4 bg-[var(--oh-surface)] p-8 text-center text-[var(--oh-muted)]">
        <Rocket className="h-7 w-7" aria-hidden />
        <div className="max-w-sm space-y-1">
          <p className="text-sm font-medium text-[var(--oh-surface-foreground)]">
            {t(I18nKey.CONVERSATION_PANEL$PREVIEW_NO_ENTRYPOINT)}
          </p>
          <p className="text-xs leading-5">
            {t(I18nKey.CONVERSATION_PANEL$PREVIEW_NO_ENTRYPOINT_DETAIL)}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0 w-full flex-col bg-[var(--oh-surface)] text-[var(--oh-surface-foreground)]">
      <div className="flex min-h-10 shrink-0 items-center gap-2 border-b border-[var(--oh-border)] px-3">
        <span
          className="min-w-0 flex-1 truncate text-xs text-[var(--oh-muted)]"
          title={entrypoint}
        >
          {entrypoint}
        </span>
        <span
          className={cn(
            "flex items-center gap-1.5 text-[11px]",
            loadError
              ? "text-red-400"
              : loaded
                ? "text-emerald-400"
                : "text-[var(--oh-muted)]",
          )}
          data-testid="live-preview-status"
        >
          <span className="h-1.5 w-1.5 rounded-full bg-current" aria-hidden />
          {loadError
            ? t(I18nKey.CONVERSATION_PANEL$PREVIEW_ERROR)
            : loaded
              ? t(I18nKey.CONVERSATION_PANEL$PREVIEW_LIVE)
              : t(I18nKey.CONVERSATION_PANEL$PREVIEW_LOADING)}
        </span>
        <button
          type="button"
          className="rounded p-1.5 text-[var(--oh-muted)] transition-colors hover:bg-[var(--oh-interactive-hover)] hover:text-[var(--oh-surface-foreground)]"
          onClick={refreshPreview}
          aria-label={t(I18nKey.CONVERSATION_PANEL$PREVIEW_REFRESH)}
          data-testid="live-preview-refresh"
        >
          <RefreshCw className="h-3.5 w-3.5" aria-hidden />
        </button>
        <button
          type="button"
          className="rounded p-1.5 text-[var(--oh-muted)] transition-colors hover:bg-[var(--oh-interactive-hover)] hover:text-[var(--oh-surface-foreground)]"
          onClick={openPreview}
          aria-label={t(I18nKey.CONVERSATION_PANEL$PREVIEW_OPEN_NEW_WINDOW)}
          data-testid="live-preview-open"
        >
          <ExternalLink className="h-3.5 w-3.5" aria-hidden />
        </button>
      </div>
      <div className="relative min-h-0 flex-1 bg-white">
        {!loaded && !loadError && (
          <div className="absolute inset-0 z-10 flex items-center justify-center bg-[var(--oh-surface)] text-xs text-[var(--oh-muted)]">
            <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden />
            {t(I18nKey.CONVERSATION_PANEL$PREVIEW_LOADING)}
          </div>
        )}
        {loadError && (
          <div className="absolute inset-0 z-10 flex items-center justify-center bg-[var(--oh-surface)] p-6 text-center text-xs text-[var(--oh-muted)]">
            {t(I18nKey.CONVERSATION_PANEL$PREVIEW_ERROR_DETAIL)}
          </div>
        )}
        <iframe
          key={previewUrl}
          title={t(I18nKey.CONVERSATION_PANEL$PREVIEW_TITLE)}
          src={previewUrl}
          className="h-full w-full border-0"
          sandbox="allow-forms allow-modals allow-popups allow-presentation allow-same-origin allow-scripts"
          onLoad={() => {
            setLoaded(true);
            setLoadError(false);
          }}
          onError={() => {
            setLoaded(false);
            setLoadError(true);
          }}
          data-testid="live-preview-frame"
        />
      </div>
    </div>
  );
}
