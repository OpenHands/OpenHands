import { useTranslation } from "react-i18next";

import type { MarsAuthState } from "#/api/mars/mars-tunnel-backend";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

interface SignInStorageNoteProps {
  authState: MarsAuthState | undefined;
  className?: string;
}

/** One line saying where the DigitalOcean sign-in is kept, so a restart holds no surprise. */
export function SignInStorageNote({
  authState,
  className,
}: SignInStorageNoteProps) {
  const { t } = useTranslation("openhands");
  const storage = authState?.storage;
  const isSaved = storage
    ? storage.kind !== "memory"
    : authState?.isPersistent !== false;

  if (!isSaved || authState?.active?.isSaved === false) {
    return (
      <p
        data-testid="digitalocean-sign-in-storage"
        className={cn("text-pretty text-warning", className)}
      >
        {t(I18nKey.DO_AGENTS$NOT_SAVED)}
      </p>
    );
  }
  if (!storage) return null;

  return (
    <p
      data-testid="digitalocean-sign-in-storage"
      className={cn("truncate text-muted", className)}
    >
      {storage.kind === "encrypted-file"
        ? t(I18nKey.DO_AGENTS$SAVED_ENCRYPTED, { host: storage.host })
        : t(I18nKey.DO_AGENTS$SAVED_IN_KEYCHAIN, {
            store: storage.label,
            host: storage.host,
          })}
    </p>
  );
}
