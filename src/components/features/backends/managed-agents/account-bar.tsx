import { useTranslation } from "react-i18next";

import type {
  MarsAuthState,
  MarsConnection,
} from "#/api/mars/mars-tunnel-backend";
import { I18nKey } from "#/i18n/declaration";
import { connectionLabel } from "./managed-agents-labels";
import { SignInStorageNote } from "./sign-in-storage-note";

interface AccountBarProps {
  authState: MarsAuthState | undefined;
  activeConnection: MarsConnection;
  onSwitchConnection: (id: string) => void;
  onAddConnection: () => void;
  onSignOut: () => void;
}

/** The signed-in DigitalOcean team, with switch, add, and sign-out. */
export function AccountBar({
  authState,
  activeConnection,
  onSwitchConnection,
  onAddConnection,
  onSignOut,
}: AccountBarProps) {
  const { t } = useTranslation("openhands");
  const connections = authState?.connections ?? [];

  return (
    <div
      data-testid="managed-agents-account"
      className="flex flex-wrap items-center gap-x-3 gap-y-1 rounded-md border border-border px-3 py-2 text-xs"
    >
      {connections.length > 1 ? (
        <label className="flex min-w-0 flex-1 items-center gap-2 text-muted">
          <span className="shrink-0">
            {t(I18nKey.BACKEND$DIGITALOCEAN_SWITCH_CONNECTION)}
          </span>
          <select
            value={activeConnection.id}
            onChange={(event) => onSwitchConnection(event.target.value)}
            data-testid="managed-agents-connection-switcher"
            className="min-w-0 flex-1 cursor-pointer rounded border border-border bg-transparent px-1.5 py-0.5 text-white"
          >
            {connections.map((connection) => (
              <option key={connection.id} value={connection.id}>
                {connection.teamName ?? connection.label}
              </option>
            ))}
          </select>
        </label>
      ) : (
        <span className="flex min-w-0 flex-1 items-center gap-2 truncate text-white">
          <span
            aria-hidden
            className="size-1.5 rounded-full bg-status-success"
          />
          {connectionLabel(activeConnection)}
        </span>
      )}
      <button
        type="button"
        onClick={onAddConnection}
        data-testid="managed-agents-add-connection"
        className="cursor-pointer text-primary hover:underline"
      >
        {t(I18nKey.BACKEND$DIGITALOCEAN_ADD_CONNECTION)}
      </button>
      <button
        type="button"
        onClick={onSignOut}
        data-testid="managed-agents-sign-out"
        className="cursor-pointer text-muted hover:text-white"
      >
        {t(I18nKey.BACKEND$DIGITALOCEAN_SIGN_OUT)}
      </button>
      <SignInStorageNote authState={authState} className="w-full" />
    </div>
  );
}
