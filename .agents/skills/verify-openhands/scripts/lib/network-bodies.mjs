// Request bodies for `browser network --bodies`, with secrets taken out.
//
// The daemon keeps the body of every app-origin write request it sees, so a
// recipe can prove what the page sent (an edited field, a stored secret
// reused behind its placeholder) without a mock. Nothing secret survives the
// redaction: everything under a key that names a credential or an `env` or
// `headers` map (strings, nested objects, arrays) is replaced by its length;
// the settings API's own `**********` placeholder is kept, so a recipe can
// tell "the placeholder was sent" from "a real value was sent".

const SECRET_KEY = /key|token|secret|auth|pass|session|sig|credential/i;
const SECRET_MAP = /^(env|headers|environment)$/i;
export const PLACEHOLDER = "**********";
export const BODY_LIMIT = 2000;

export function redactBodyValue(value) {
  if (typeof value !== "string" || value === "" || value === PLACEHOLDER)
    return value;
  return `<redacted ${value.length} chars>`;
}

// `secret` is inherited by the whole subtree: a string in an array under
// `api_keys`, an object under `credentials`, every value of an `env` map.
function redactNode(node, secret = false) {
  if (typeof node === "string") return secret ? redactBodyValue(node) : node;
  if (Array.isArray(node)) return node.map((v) => redactNode(v, secret));
  if (node && typeof node === "object") {
    const out = {};
    for (const [key, value] of Object.entries(node))
      out[key] = redactNode(
        value,
        secret || SECRET_KEY.test(key) || SECRET_MAP.test(key),
      );
    return out;
  }
  return node;
}

// Returns the body as the page sent it, JSON re-serialized after redaction,
// cut at `limit` characters; a body that is not JSON is described, not shown.
// `all` redacts every string value, for endpoints whose whole payload is a
// credential (secrets, login, tokens) whatever the field names.
export const SECRET_PATH = /secret|credential|api-key|apikey|auth|login|token/i;

export function redactBody(text, { limit = BODY_LIMIT, all = false } = {}) {
  if (text === undefined || text === null || text === "") return undefined;
  const raw = String(text);
  let shown;
  try {
    shown = JSON.stringify(redactNode(JSON.parse(raw), all));
  } catch {
    if (/^[^=&\s]+=[^&]*(&[^=&\s]+=[^&]*)*$/.test(raw)) {
      const params = new URLSearchParams(raw);
      for (const key of [...params.keys()])
        if (all || SECRET_KEY.test(key)) params.set(key, "<redacted>");
      shown = params.toString();
    } else {
      return `<non-JSON body, ${raw.length} chars>`;
    }
  }
  return shown.length > limit ? `${shown.slice(0, limit)}…` : shown;
}
