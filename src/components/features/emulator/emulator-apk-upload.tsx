import { useCallback, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { validateApkFile } from "#/api/pentest/mobile-artifacts-service";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { LoadingSpinner } from "#/components/shared/loading-spinner";

type EmulatorApkUploadProps = {
  uploading: boolean;
  errorKey?: "ipa" | "type" | "size" | "failed" | null;
  offlineHint?: boolean;
  onFileAccepted: (file: File) => void;
};

function rejectKey(reason: "ipa" | "type" | "size"): I18nKey {
  if (reason === "ipa") return I18nKey.EMULATOR$UPLOAD_REJECT_IPA;
  if (reason === "size") return I18nKey.EMULATOR$UPLOAD_REJECT_SIZE;
  return I18nKey.EMULATOR$UPLOAD_REJECT_TYPE;
}

export function EmulatorApkUpload({
  uploading,
  errorKey,
  offlineHint = false,
  onFileAccepted,
}: EmulatorApkUploadProps) {
  const { t } = useTranslation("openhands");
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [localError, setLocalError] = useState<I18nKey | null>(null);

  const handleFile = useCallback(
    (file: File | undefined | null) => {
      if (!file) return;
      const result = validateApkFile(file);
      if (!result.ok) {
        setLocalError(rejectKey(result.reason));
        return;
      }
      setLocalError(null);
      onFileAccepted(file);
    },
    [onFileAccepted],
  );

  const displayError =
    localError ??
    (errorKey === "ipa"
      ? I18nKey.EMULATOR$UPLOAD_REJECT_IPA
      : errorKey === "type"
        ? I18nKey.EMULATOR$UPLOAD_REJECT_TYPE
        : errorKey === "size"
          ? I18nKey.EMULATOR$UPLOAD_REJECT_SIZE
          : errorKey === "failed"
            ? I18nKey.EMULATOR$UPLOAD_FAILED
            : null);

  return (
    <div className="flex flex-col gap-2">
      <button
        type="button"
        data-testid="emulator-apk-dropzone"
        aria-label={t(I18nKey.EMULATOR$UPLOAD_DROPZONE)}
        disabled={uploading}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            inputRef.current?.click();
          }
        }}
        onDragOver={(event) => {
          event.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragOver(false);
          const file = event.dataTransfer.files?.[0];
          handleFile(file);
        }}
        className={cn(
          "flex min-h-11 w-full flex-col items-center justify-center gap-1 rounded border border-dashed px-3 py-3 text-center",
          "border-[var(--oh-border)] text-sm text-[var(--oh-muted)]",
          "cursor-pointer focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2",
          dragOver && "border-[var(--foreground)] text-[var(--foreground)]",
          uploading && "cursor-wait opacity-70",
        )}
      >
        <span>{t(I18nKey.EMULATOR$UPLOAD_DROPZONE)}</span>
        <span className="text-xs">{t(I18nKey.EMULATOR$UPLOAD_ACCEPT)}</span>
      </button>
      <input
        ref={inputRef}
        type="file"
        accept=".apk,application/vnd.android.package-archive"
        className="hidden"
        data-testid="emulator-apk-input"
        onChange={(event) => {
          const file = event.target.files?.[0];
          handleFile(file);
          if (inputRef.current) {
            inputRef.current.value = "";
          }
        }}
      />
      {offlineHint && (
        <p className="text-xs text-[var(--oh-muted)]">
          {t(I18nKey.EMULATOR$UPLOAD_HINT_OFFLINE)}
        </p>
      )}
      {uploading && (
        <div
          className="flex items-center gap-2 text-xs text-[var(--oh-muted)]"
          role="status"
          aria-live="polite"
          data-testid="emulator-upload-progress"
        >
          <LoadingSpinner size="small" />
          <span>{t(I18nKey.EMULATOR$UPLOAD_PROGRESS)}</span>
        </div>
      )}
      {displayError && (
        <p
          className="text-xs text-red-400"
          role="alert"
          aria-live="assertive"
          data-testid="emulator-upload-error"
        >
          {t(displayError)}
        </p>
      )}
    </div>
  );
}
