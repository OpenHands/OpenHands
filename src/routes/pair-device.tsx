import React from "react";
import { useTranslation } from "react-i18next";
import {
  captureDevicePairingSecret,
  redeemDevicePairing,
} from "#/api/mars/mars-device-pairing";
import { getRegisteredBackends } from "#/api/backend-registry/active-store";
import { DEFAULT_LOCAL_BACKEND_NAME } from "#/api/backend-registry/default-backend";
import { useActiveBackendContext } from "#/contexts/active-backend-context";
import { I18nKey } from "#/i18n/declaration";
import { buildAgentCanvasPath } from "#/utils/base-path";

/** Public landing page: possession of the fragment is the one-time proof. */
export default function PairDevice() {
  const { t } = useTranslation("openhands");
  const { active, addBackend, updateBackend, setActive } =
    useActiveBackendContext();
  const [secret, setSecret] = React.useState<string | null>(null);
  const [busy, setBusy] = React.useState(false);
  const [failed, setFailed] = React.useState(false);

  React.useEffect(() => {
    setSecret(captureDevicePairingSecret());
  }, []);

  const connect = async () => {
    if (!secret || busy) return;
    setBusy(true);
    try {
      const { sessionApiKey } = await redeemDevicePairing(secret);
      const backends = getRegisteredBackends();
      const host = window.location.origin;
      const existing = backends.find(
        (item) =>
          item.kind === "local" &&
          !item.marsSessionId &&
          item.host.replace(/\/+$/, "") === host,
      );
      if (existing) updateBackend(existing.id, { apiKey: sessionApiKey });
      else {
        addBackend({
          name: DEFAULT_LOCAL_BACKEND_NAME,
          kind: "local",
          host,
          apiKey: sessionApiKey,
        });
        if (backends.length > 0) setActive(active.backend.id, active.orgId);
      }
      window.location.replace(buildAgentCanvasPath("/"));
    } catch {
      setFailed(true);
      setBusy(false);
    }
  };

  return (
    <main className="flex min-h-screen items-center justify-center bg-canvas-base p-6 text-contrast">
      <div className="flex max-w-md flex-col gap-4 rounded-xl border border-border bg-base-secondary p-6">
        <h1 className="text-lg font-semibold">
          {t(I18nKey.BACKEND$PAIR_DEVICE)}
        </h1>
        <p className="text-sm text-muted">
          {t(I18nKey.BACKEND$PAIR_DEVICE_ACCESS)}
        </p>
        {failed || !secret ? (
          <p role="alert" className="text-sm text-status-error">
            {t(I18nKey.BACKEND$PAIR_DEVICE_FAILED)}
          </p>
        ) : null}
        <button
          type="button"
          disabled={!secret || busy || failed}
          onClick={() => void connect()}
          className="cursor-pointer rounded bg-primary px-4 py-2 text-black disabled:cursor-not-allowed disabled:opacity-50"
        >
          {t(I18nKey.BACKEND$PAIR_DEVICE_CONNECT)}
        </button>
      </div>
    </main>
  );
}
