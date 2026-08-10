import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  EmulatorRequestError,
  EmulatorService,
} from "#/api/integrations/emulator-service";
import MobileArtifactsService from "#/api/pentest/mobile-artifacts-service";
import type { MobileArtifact } from "#/api/pentest/mobile-artifacts-types";
import { getStoredConversationMetadata } from "#/api/conversation-metadata-store";
import { EmulatorEmptyState } from "#/components/features/emulator/emulator-empty-state";
import { EmulatorToolbar } from "#/components/features/emulator/emulator-toolbar";
import { EmulatorApkUpload } from "#/components/features/emulator/emulator-apk-upload";
import { EmulatorArtifactsList } from "#/components/features/emulator/emulator-artifacts-list";
import { useConversationId } from "#/hooks/use-conversation-id";
import { MOBILE_ARTIFACTS_QUERY_KEYS } from "#/hooks/query/query-keys";
import { I18nKey } from "#/i18n/declaration";

type EmulatorViewState =
  | { kind: "loading" }
  | { kind: "idle"; unavailable: boolean }
  | { kind: "starting" }
  | { kind: "live"; url: string; iframeKey: number }
  | { kind: "error"; message: string; unavailable?: boolean };

const IFRAME_SANDBOX =
  "allow-scripts allow-same-origin allow-forms allow-popups allow-downloads";

export function EmulatorPanel() {
  const { t } = useTranslation("openhands");
  const { conversationId } = useConversationId();
  const queryClient = useQueryClient();
  const [view, setView] = useState<EmulatorViewState>({ kind: "loading" });
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<
    "ipa" | "type" | "size" | "failed" | null
  >(null);
  const [installBusy, setInstallBusy] = useState(false);

  const engagementId =
    getStoredConversationMetadata(conversationId)?.engagement_id ?? null;

  const artifactsQuery = useQuery({
    queryKey: engagementId
      ? MOBILE_ARTIFACTS_QUERY_KEYS.list(engagementId)
      : MOBILE_ARTIFACTS_QUERY_KEYS.all,
    queryFn: () => MobileArtifactsService.listArtifacts(engagementId!),
    enabled: Boolean(engagementId),
    refetchInterval: (query) => {
      const items = query.state.data?.items ?? [];
      const pending = items.some(
        (item) =>
          item.scan_status === "queued" || item.scan_status === "scanning",
      );
      return pending ? 1500 : false;
    },
    meta: { disableToast: true },
  });

  const artifacts: MobileArtifact[] = artifactsQuery.data?.items ?? [];

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const status = await EmulatorService.getStatus();
        if (cancelled) return;
        if (status.ready) {
          setView({
            kind: "live",
            url: status.url || EmulatorService.iframePath(),
            iframeKey: 0,
          });
          return;
        }
        setView({ kind: "idle", unavailable: status.unavailable });
      } catch {
        if (!cancelled) {
          setView({ kind: "idle", unavailable: true });
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const startEmulator = useCallback(async () => {
    setView({ kind: "starting" });
    try {
      const status = await EmulatorService.start();
      if (!status.ready) {
        setView({
          kind: "error",
          message: status.unavailable
            ? t(I18nKey.EMULATOR$UNAVAILABLE)
            : t(I18nKey.EMULATOR$FAILED),
          unavailable: status.unavailable,
        });
        return;
      }
      setView({
        kind: "live",
        url: status.url || EmulatorService.iframePath(),
        iframeKey: 0,
      });
    } catch (err) {
      const unavailable =
        err instanceof EmulatorRequestError ? err.unavailable : false;
      setView({
        kind: "error",
        message: unavailable
          ? t(I18nKey.EMULATOR$UNAVAILABLE)
          : t(I18nKey.EMULATOR$FAILED),
        unavailable,
      });
    }
  }, [t]);

  const refreshIframe = useCallback(() => {
    setView((prev) =>
      prev.kind === "live" ? { ...prev, iframeKey: prev.iframeKey + 1 } : prev,
    );
  }, []);

  const handleFileAccepted = useCallback(
    async (file: File) => {
      if (!engagementId) {
        setUploadError("failed");
        return;
      }
      setUploadError(null);
      setUploading(true);
      try {
        await MobileArtifactsService.uploadApk(engagementId, file);
        await queryClient.invalidateQueries({
          queryKey: MOBILE_ARTIFACTS_QUERY_KEYS.list(engagementId),
        });
      } catch {
        setUploadError("failed");
      } finally {
        setUploading(false);
      }
    },
    [engagementId, queryClient],
  );

  const handleInstall = useCallback(
    async (artifactId: string) => {
      if (!engagementId || view.kind !== "live") return;
      const confirmed = window.confirm(t(I18nKey.EMULATOR$INSTALL_CONFIRM));
      if (!confirmed) return;
      setInstallBusy(true);
      try {
        await MobileArtifactsService.installArtifact(engagementId, artifactId);
      } finally {
        setInstallBusy(false);
      }
    },
    [engagementId, t, view.kind],
  );

  const stageUnavailable =
    (view.kind === "idle" && view.unavailable) ||
    (view.kind === "error" && view.unavailable);
  // Controlled disclosure: closed by default in live, open otherwise.
  // Reset only on live↔rest; refetch must not remount/close user toggle.
  const railDefaultOpen = view.kind !== "live";
  const [railOpen, setRailOpen] = useState(railDefaultOpen);
  const prevRailDefaultOpen = useRef(railDefaultOpen);

  useEffect(() => {
    if (prevRailDefaultOpen.current !== railDefaultOpen) {
      prevRailDefaultOpen.current = railDefaultOpen;
      setRailOpen(railDefaultOpen);
    }
  }, [railDefaultOpen]);

  return (
    <div className="flex h-full min-h-0 flex-col" data-testid="emulator-panel">
      {view.kind === "live" && <EmulatorToolbar onRefresh={refreshIframe} />}

      <div className="flex min-h-0 flex-1 flex-col">
        {view.kind === "live" ? (
          <iframe
            key={view.iframeKey}
            title={t(I18nKey.EMULATOR$IFRAME_TITLE)}
            data-testid="emulator-iframe"
            src={view.url}
            className="h-full w-full min-h-0 flex-1 border-0 bg-black"
            sandbox={IFRAME_SANDBOX}
            allow="clipboard-read; clipboard-write"
          />
        ) : (
          <EmulatorEmptyState
            kind={
              view.kind === "loading"
                ? "loading"
                : view.kind === "starting"
                  ? "starting"
                  : stageUnavailable
                    ? "unavailable"
                    : view.kind === "error"
                      ? "error"
                      : "idle"
            }
            message={view.kind === "error" ? view.message : undefined}
            onStart={
              stageUnavailable ||
              view.kind === "loading" ||
              view.kind === "starting"
                ? undefined
                : () => void startEmulator()
            }
          />
        )}
      </div>

      <details
        className="shrink-0 border-t border-[var(--oh-border)]"
        open={railOpen}
        onToggle={(event) => {
          setRailOpen(event.currentTarget.open);
        }}
        data-testid="emulator-artifacts-rail"
      >
        <summary className="cursor-pointer px-3 py-2 text-sm text-[var(--foreground)]">
          {t(I18nKey.EMULATOR$UPLOAD_SECTION)}
        </summary>
        <div className="flex max-h-[200px] flex-col gap-3 px-3 pb-3">
          <EmulatorApkUpload
            uploading={uploading}
            errorKey={uploadError}
            offlineHint={stageUnavailable || view.kind !== "live"}
            onFileAccepted={(file) => void handleFileAccepted(file)}
          />
          <EmulatorArtifactsList
            artifacts={artifacts}
            installEnabled={view.kind === "live" && !installBusy}
            onRequestInstall={(id) => void handleInstall(id)}
          />
        </div>
      </details>
    </div>
  );
}
