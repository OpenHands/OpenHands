import React from "react";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { Check, Copy, Loader2 } from "lucide-react";
import qrcode from "qrcode-generator";

import { requireMarsBridge } from "#/api/mars/mars-tunnel-backend";
import { BrandButton } from "#/components/features/settings/brand-button";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import {
  MODAL_MAX_WIDTH_VIEWPORT,
  modalWidthClassName,
} from "#/components/shared/modals/modal-body";
import { BaseModalTitle } from "#/components/shared/modals/confirmation-modals/base-modal";
import { MARS_QUERY_KEYS } from "#/hooks/query/query-keys";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

/** Blank modules around the code, which scanners need. */
const QUIET_ZONE = 4;
/**
 * Fixed, not themed: a code inverted by a light theme scans poorly on many
 * phones.
 */
const QR_DARK = "#000";
const QR_LIGHT = "#fff";

function QrCode({ value, label }: { value: string; label: string }) {
  const { size, path } = React.useMemo(() => {
    const qr = qrcode(0, "M");
    qr.addData(value);
    qr.make();
    const count = qr.getModuleCount();
    let d = "";
    for (let row = 0; row < count; row += 1) {
      for (let col = 0; col < count; col += 1) {
        if (qr.isDark(row, col)) d += `M${col} ${row}h1v1h-1z`;
      }
    }
    return { size: count, path: d };
  }, [value]);
  const box = size + QUIET_ZONE * 2;

  return (
    <svg
      data-testid="add-device-qr"
      role="img"
      aria-label={label}
      viewBox={`${-QUIET_ZONE} ${-QUIET_ZONE} ${box} ${box}`}
      shapeRendering="crispEdges"
      className="size-52 rounded-lg"
    >
      <rect
        x={-QUIET_ZONE}
        y={-QUIET_ZONE}
        width={box}
        height={box}
        fill={QR_LIGHT}
      />
      <path d={path} fill={QR_DARK} />
    </svg>
  );
}

const isLoopbackHostname = (hostname: string) =>
  hostname === "localhost" ||
  hostname === "[::1]" ||
  /^127\.\d+\.\d+\.\d+$/.test(hostname);

/** A link the other device would follow over the network without encryption. */
const isPlainHttpOnNetwork = (link: string) => {
  try {
    const url = new URL(link);
    return url.protocol === "http:" && !isLoopbackHostname(url.hostname);
  } catch {
    return false;
  }
};

interface AddDeviceDialogProps {
  onClose: () => void;
}

/**
 * "Add another device": a one-time link, shown as a QR code, that lets
 * another device in with the same access as typing the Canvas key.
 */
export default function AddDeviceDialog({ onClose }: AddDeviceDialogProps) {
  const { t } = useTranslation("openhands");
  const [attempt, setAttempt] = React.useState(0);
  const [selected, setSelected] = React.useState(0);
  const [copyState, setCopyState] = React.useState<
    "idle" | "copied" | "failed"
  >("idle");
  const [isExpired, setIsExpired] = React.useState(false);
  const linkField = React.useRef<HTMLInputElement>(null);
  const pairing = useQuery({
    queryKey: MARS_QUERY_KEYS.pairingLink(attempt),
    queryFn: () => requireMarsBridge().createPairingLink!(),
    staleTime: Infinity,
    gcTime: 0,
    retry: false,
    refetchOnWindowFocus: false,
  });
  const links = pairing.data?.links ?? [];
  const link = links[Math.min(selected, links.length - 1)] ?? null;

  // The link stops working at its expiry; say so instead of showing a dead code.
  React.useEffect(() => {
    const expiresAt = pairing.data?.expiresAt;
    if (!expiresAt) return undefined;
    const timer = window.setTimeout(
      () => setIsExpired(true),
      Math.max(0, Date.parse(expiresAt) - Date.now()),
    );
    return () => window.clearTimeout(timer);
  }, [pairing.data?.expiresAt]);

  const newLink = () => {
    setIsExpired(false);
    setCopyState("idle");
    setAttempt((value) => value + 1);
  };

  const copy = async () => {
    if (!link) return;
    try {
      // Browsers keep the clipboard to HTTPS and localhost, so on a plain
      // HTTP address `navigator.clipboard` is missing and this throws.
      await navigator.clipboard.writeText(link);
      setCopyState("copied");
    } catch {
      linkField.current?.focus();
      linkField.current?.select();
      setCopyState("failed");
    }
  };

  let body: React.ReactNode;
  if (pairing.isPending) {
    body = (
      <div className="flex h-52 items-center justify-center">
        <Loader2 className="size-5 animate-spin text-muted" aria-hidden />
      </div>
    );
  } else if (pairing.isError) {
    body = (
      <p role="alert" className="text-sm text-status-error">
        {pairing.error instanceof Error
          ? pairing.error.message
          : t(I18nKey.DEVICES$LINK_FAILED)}
      </p>
    );
  } else if (!link) {
    body = (
      <p
        data-testid="add-device-loopback-only"
        className="text-pretty text-sm text-muted"
      >
        {t(I18nKey.DEVICES$LOOPBACK_ONLY)}
      </p>
    );
  } else if (isExpired) {
    body = (
      <div className="flex flex-col items-center gap-3">
        <p data-testid="add-device-expired" className="text-sm text-muted">
          {t(I18nKey.DEVICES$EXPIRED)}
        </p>
        <BrandButton
          type="button"
          variant="primary"
          testId="add-device-new-link"
          onClick={newLink}
        >
          {t(I18nKey.DEVICES$NEW_LINK)}
        </BrandButton>
      </div>
    );
  } else {
    body = (
      <div className="flex flex-col items-center gap-3">
        <QrCode value={link} label={t(I18nKey.DEVICES$QR_LABEL)} />
        {links.length > 1 ? (
          <label className="flex w-full items-center gap-2 text-xs text-muted">
            <span className="shrink-0">{t(I18nKey.DEVICES$ADDRESS)}</span>
            <select
              value={Math.min(selected, links.length - 1)}
              onChange={(event) => {
                setSelected(Number(event.target.value));
                setCopyState("idle");
              }}
              data-testid="add-device-address"
              className="min-w-0 flex-1 cursor-pointer rounded border border-border bg-transparent px-1.5 py-0.5 text-white"
            >
              {links.map((candidate, index) => (
                <option key={candidate} value={index}>
                  {new URL(candidate).host}
                </option>
              ))}
            </select>
          </label>
        ) : null}
        <input
          ref={linkField}
          readOnly
          value={link}
          aria-label={t(I18nKey.DEVICES$LINK_LABEL)}
          data-testid="add-device-link"
          onFocus={(event) => event.currentTarget.select()}
          className="w-full rounded border border-border bg-transparent px-2 py-1 font-mono text-xs text-white"
        />
        <BrandButton
          type="button"
          variant="secondary"
          testId="add-device-copy"
          onClick={() => void copy()}
          className="w-full"
          startContent={
            copyState === "copied" ? (
              <Check className="size-4" aria-hidden />
            ) : (
              <Copy className="size-4" aria-hidden />
            )
          }
        >
          {t(
            copyState === "copied"
              ? I18nKey.DEVICES$COPIED
              : I18nKey.DEVICES$COPY_LINK,
          )}
        </BrandButton>
        {copyState === "failed" ? (
          <p
            role="alert"
            data-testid="add-device-copy-failed"
            className="text-pretty text-center text-xs text-status-error"
          >
            {t(I18nKey.DEVICES$COPY_FAILED)}
          </p>
        ) : null}
        <p className="text-center text-xs text-muted">
          {t(I18nKey.DEVICES$WORKS_ONCE)}
        </p>
        {isPlainHttpOnNetwork(link) ? (
          <p
            data-testid="add-device-plain-http"
            className="text-pretty text-xs text-warning"
          >
            {t(I18nKey.DEVICES$PLAIN_HTTP)}
          </p>
        ) : null}
      </div>
    );
  }

  return (
    <ModalBackdrop
      onClose={onClose}
      aria-label={t(I18nKey.DEVICES$ADD_ANOTHER)}
    >
      <div
        data-testid="add-device-dialog"
        className={cn(
          "flex flex-col gap-4 rounded-xl border border-border bg-base-secondary p-5",
          modalWidthClassName("sm"),
          MODAL_MAX_WIDTH_VIEWPORT,
        )}
      >
        <div className="flex flex-col gap-1">
          <BaseModalTitle title={t(I18nKey.DEVICES$ADD_ANOTHER)} />
          <p className="text-pretty text-sm text-muted">
            {t(I18nKey.DEVICES$EXPLAIN)}
          </p>
        </div>
        {body}
        <BrandButton
          type="button"
          variant="secondary"
          testId="add-device-close"
          onClick={onClose}
        >
          {t(I18nKey.BUTTON$CLOSE)}
        </BrandButton>
      </div>
    </ModalBackdrop>
  );
}
