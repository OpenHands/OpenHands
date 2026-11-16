import { useState } from "react";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useCloudSandbox } from "#/hooks/query/use-cloud-sandbox";
import { I18nKey } from "#/i18n/declaration";
import { ErrorMessageBanner } from "./error-message-banner";

const SANDBOX_POLL_INTERVAL_MS = 30_000;

export function SandboxOomNotice({
  sandboxId,
}: {
  sandboxId: string | null | undefined;
}) {
  const { backend, orgId } = useActiveBackend();
  const { data: sandbox } = useCloudSandbox(
    sandboxId,
    SANDBOX_POLL_INTERVAL_MS,
  );
  const [dismissedIncident, setDismissedIncident] = useState<string | null>(
    null,
  );
  const termination = sandbox?.last_termination;
  const incident = JSON.stringify([
    backend.id,
    orgId,
    sandboxId,
    termination?.finished_at,
  ]);

  if (
    backend.kind !== "cloud" ||
    termination?.reason !== "OOMKilled" ||
    incident === dismissedIncident
  ) {
    return null;
  }

  return (
    <ErrorMessageBanner
      message={I18nKey.CHAT_INTERFACE$SANDBOX_OUT_OF_MEMORY}
      onDismiss={() => setDismissedIncident(incident)}
    />
  );
}
