import React from "react";
import { QRCodeSVG } from "qrcode.react";
import { useTranslation } from "react-i18next";
import {
  buildDevicePairingLink,
  cancelDevicePairing,
  createDevicePairing,
  isUnencryptedNetworkUrl,
  type DevicePairingInvitation,
} from "#/api/mars/mars-device-pairing";
import { I18nKey } from "#/i18n/declaration";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalBody } from "#/components/shared/modals/modal-body";

const EXPIRY_CHECK_MS = 1_000;

/** Only mounts while open, so closing revokes the invitation and clears its QR. */
function PairingDialog({
  invitation,
  onClose,
}: {
  invitation: DevicePairingInvitation;
  onClose: () => void;
}) {
  const { t } = useTranslation("openhands");
  const [origin, setOrigin] = React.useState(window.location.origin);
  const [expired, setExpired] = React.useState(
    () => Date.now() >= invitation.expiresAt,
  );
  const [copied, setCopied] = React.useState(false);

  React.useEffect(() => {
    const timer = setInterval(
      () => setExpired(Date.now() >= invitation.expiresAt),
      EXPIRY_CHECK_MS,
    );
    return () => clearInterval(timer);
  }, [invitation]);

  let link: string | null = null;
  try {
    if (invitation && !expired)
      link = buildDevicePairingLink(origin, invitation.secret);
  } catch {
    /* Keep the address editable; never create a non-HTTP link. */
  }

  return (
    <ModalBackdrop
      onClose={onClose}
      aria-label={t(I18nKey.BACKEND$PAIR_DEVICE)}
    >
      <ModalBody
        width="md"
        // eslint-disable-next-line shadcn/no-unknown-classes -- PostHog consumes this opt-out marker; it deliberately has no CSS.
        className="ph-no-capture max-h-[90dvh] max-w-[90vw] items-stretch gap-4 overflow-y-auto"
      >
        <h2 className="text-lg font-semibold">
          {t(I18nKey.BACKEND$PAIR_DEVICE)}
        </h2>
        <p className="text-sm text-muted">
          {t(I18nKey.BACKEND$PAIR_DEVICE_HELP)}
        </p>
        <label className="flex flex-col gap-1 text-sm">
          {t(I18nKey.BACKEND$PAIR_DEVICE_ADDRESS)}
          <input
            className="rounded border border-border bg-canvas-base p-2 text-contrast"
            value={origin}
            onChange={(event) => {
              setOrigin(event.target.value);
              setCopied(false);
            }}
          />
        </label>
        {isUnencryptedNetworkUrl(origin) ? (
          <p role="alert" className="text-sm text-feedback-warning">
            {t(I18nKey.BACKEND$UNENCRYPTED_CONNECTION)}
          </p>
        ) : null}
        {link ? (
          <>
            <div className="self-center rounded bg-white p-3">
              <QRCodeSVG
                value={link}
                size={192}
                marginSize={2}
                title={t(I18nKey.BACKEND$PAIR_DEVICE)}
              />
            </div>
            <input
              aria-label={t(I18nKey.BACKEND$PAIR_DEVICE_LINK)}
              readOnly
              value={link}
              className="w-full rounded border border-border bg-canvas-base p-2 text-xs text-contrast"
              onFocus={(event) => event.target.select()}
            />
            <button
              type="button"
              className="cursor-pointer text-primary hover:underline"
              onClick={() => {
                void navigator.clipboard
                  ?.writeText(link)
                  .then(() => setCopied(true))
                  .catch(() => setCopied(false));
              }}
            >
              {t(copied ? I18nKey.BUTTON$COPIED : I18nKey.BUTTON$COPY)}
            </button>
          </>
        ) : null}
        {expired ? (
          <p role="alert" className="text-sm text-status-error">
            {t(I18nKey.BACKEND$PAIR_DEVICE_EXPIRED)}
          </p>
        ) : null}
        <button
          type="button"
          onClick={onClose}
          className="cursor-pointer self-end text-muted hover:text-contrast"
        >
          {t(I18nKey.BUTTON$CLOSE)}
        </button>
      </ModalBody>
    </ModalBackdrop>
  );
}

export function DevicePairingButton() {
  const { t } = useTranslation("openhands");
  const [invitation, setInvitation] =
    React.useState<DevicePairingInvitation | null>(null);
  const [pending, setPending] = React.useState(false);
  const [failed, setFailed] = React.useState(false);
  const create = async () => {
    setPending(true);
    setFailed(false);
    try {
      setInvitation(await createDevicePairing());
    } catch {
      setFailed(true);
    } finally {
      setPending(false);
    }
  };
  const close = () => {
    if (invitation) void cancelDevicePairing(invitation.secret).catch(() => {});
    setInvitation(null);
  };
  return (
    <>
      <button
        type="button"
        className="shrink-0 cursor-pointer text-primary hover:underline"
        disabled={pending}
        onClick={() => void create()}
      >
        {t(I18nKey.BACKEND$PAIR_DEVICE)}
      </button>
      {failed ? (
        <span role="alert" className="text-status-error">
          {t(I18nKey.BACKEND$PAIR_DEVICE_FAILED)}
        </span>
      ) : null}
      {invitation ? (
        <PairingDialog invitation={invitation} onClose={close} />
      ) : null}
    </>
  );
}
