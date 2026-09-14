import { useMemo } from "react";

import { useActiveBackend } from "#/contexts/active-backend-context";
import { useCloudSandbox } from "#/hooks/query/use-cloud-sandbox";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useRuntimeIsReady } from "#/hooks/use-runtime-is-ready";

const WORKER_URL_PREFIX = "WORKER_";

function isBrowserUrl(value: string | undefined): value is string {
  if (!value) return false;
  try {
    const url = new URL(value);
    return url.protocol === "http:" || url.protocol === "https:";
  } catch {
    return false;
  }
}

export function chooseWorkerUrl(
  exposedUrls: Array<{ name: string; url: string }> | null | undefined,
): string | null {
  const workerUrl = exposedUrls
    ?.filter(
      (entry) =>
        entry.name.startsWith(WORKER_URL_PREFIX) && isBrowserUrl(entry.url),
    )
    .sort((left, right) => left.name.localeCompare(right.name))[0]?.url;
  return workerUrl ?? null;
}

export interface ForwardedPreviewUrl {
  url: string | null;
  isLoading: boolean;
  isError: boolean;
  isForwarded: boolean;
  refetch: () => Promise<void>;
}

/**
 * Resolves the real browser URL for an application server.
 *
 * Cloud sandboxes already expose a first-class `exposed_urls` contract. The
 * Agent Server turns URLs named `WORKER_*` into the agent's work-host skill;
 * this hook consumes that same contract for the GUI. Local conversations keep
 * using the validated workspace fileserver path because local Agent Server has
 * no Cloud sandbox metadata to query.
 */
export function useForwardedPreviewUrl(): ForwardedPreviewUrl {
  const active = useActiveBackend();
  const { data: conversation } = useActiveConversation();
  const runtimeIsReady = useRuntimeIsReady();
  const isCloud = active.backend.kind === "cloud";
  const sandboxQuery = useCloudSandbox(
    isCloud ? conversation?.sandbox_id : null,
    { pollForWorkerUrls: isCloud && runtimeIsReady },
  );

  const url = useMemo(
    () => chooseWorkerUrl(sandboxQuery.data?.exposed_urls),
    [sandboxQuery.data?.exposed_urls],
  );

  const waitingForWorkerUrl =
    isCloud &&
    !url &&
    !sandboxQuery.isError &&
    (!sandboxQuery.data ||
      sandboxQuery.data.status === "STARTING" ||
      sandboxQuery.data.status === "RUNNING");

  return {
    url,
    isLoading: waitingForWorkerUrl,
    isError: isCloud && sandboxQuery.isError,
    isForwarded: isCloud && !!url,
    refetch: async () => {
      await sandboxQuery.refetch();
    },
  };
}
