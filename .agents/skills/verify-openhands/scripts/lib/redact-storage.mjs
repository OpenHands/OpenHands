// Web storage values for `browser storage --values`, safe to paste into
// reports. A value stored under a secret-looking key (Canvas keeps the
// transcription API key as a plain string under
// `openhands-transcription-api-key`) is redacted whole; JSON values (backend
// entries embed session keys) have their secret-looking fields redacted.
const SECRET_STORAGE_KEY =
  /api[-_]?key|session[-_]?key|token|secret|password|credential/i;
const SECRET_FIELD = /key|token|secret|password|auth/i;
const REDACTED = "<redacted>";

export function redactStorageValue(storageKey, value) {
  let parsed;
  try {
    parsed = JSON.parse(value);
  } catch {
    parsed = undefined;
  }
  if (parsed !== null && typeof parsed === "object") {
    return JSON.stringify(parsed, (field, v) =>
      SECRET_FIELD.test(field) && typeof v === "string" && v ? REDACTED : v,
    );
  }
  return SECRET_STORAGE_KEY.test(storageKey) && value ? REDACTED : value;
}

export function redactStorage(entries) {
  return Object.fromEntries(
    Object.entries(entries).map(([k, v]) => [k, redactStorageValue(k, v)]),
  );
}
