// Web storage values for `browser storage --values`, with secret-looking
// values hidden by name. Storage holds only strings, and each value is shown
// as follows:
// - Under a storage key that matches SECRET_NAME, a non-empty value becomes
//   "<redacted>" whole, whatever it holds: a plain string (Canvas keeps the
//   transcription API key under `openhands-transcription-api-key`), or a JSON
//   string, array, object or number.
// - Under any other key, a JSON array or object (backend entries embed API
//   keys) keeps its structure, but at any depth a field whose name matches
//   SECRET_NAME has its value replaced by "<redacted>" when that value is a
//   number or a non-empty string, array or object. A container is replaced
//   whole, property names included. Booleans, null and empty values stay, so
//   `"authenticated": true` stays readable.
// - Anything else (an empty value, a plain string, or a JSON string, number,
//   boolean or null) is returned unchanged.
// Storage keys and JSON fields share one pattern. Redaction goes by name only:
// a secret under an innocent name, in free text such as a composer draft, or
// in JSON stored inside a JSON string is printed. Read the output before
// quoting it (report.md: do not publish localStorage).
const SECRET_NAME = /key|token|secret|pass|auth|credential/i;
const REDACTED = "<redacted>";

// True for a number or a non-empty string, array or object.
function hasContent(v) {
  if (typeof v === "number") return true;
  if (typeof v === "string") return v !== "";
  return v !== null && typeof v === "object" && Object.keys(v).length > 0;
}

export function redactStorageValue(storageKey, value) {
  if (!value) return value;
  if (SECRET_NAME.test(storageKey)) return REDACTED;
  let parsed;
  try {
    parsed = JSON.parse(value);
  } catch {
    return value;
  }
  if (parsed === null || typeof parsed !== "object") return value;
  return JSON.stringify(parsed, (field, v) =>
    SECRET_NAME.test(field) && hasContent(v) ? REDACTED : v,
  );
}

export function redactStorage(entries) {
  return Object.fromEntries(
    Object.entries(entries).map(([k, v]) => [k, redactStorageValue(k, v)]),
  );
}
