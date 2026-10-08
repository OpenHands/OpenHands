// Request bodies for `browser network --bodies`, with secrets taken out.
//
// The daemon keeps the body of every app-origin write request it sees, so a
// recipe can prove what the page sent (an edited field, a stored secret
// reused behind its placeholder) without a mock. Nothing secret survives the
// redaction: everything under a key that names a credential or an `env` or
// `headers` map (strings, nested objects, arrays) is replaced by its length;
// the settings API's own `**********` placeholder is kept, so a recipe can
// tell "the placeholder was sent" from "a real value was sent". A URL in any
// other string (an MCP server's `url`, a prompt) keeps its host and path but
// loses its userinfo and the values of its secret-named query parameters.

// Matched as a substring, so a non-secret key such as `design` (sig),
// `keyboard` (key) or `bypass` (pass) reads `<redacted N chars>` too; over-
// redaction is the safe side. Shared with the daemon's query redaction.
export const SECRET_KEY = /key|token|secret|auth|pass|session|sig|credential/i;
const SECRET_MAP = /^(env|headers|environment)$/i;
export const PLACEHOLDER = "**********";
export const BODY_LIMIT = 2000;

export function redactBodyValue(value) {
  if (typeof value !== "string" || value === "" || value === PLACEHOLDER)
    return value;
  return `<redacted ${value.length} chars>`;
}

// A URL carries credentials of its own, under no key at all: the userinfo
// of `https://user:pw@host/` and the values of `?api_key=…`-style query
// parameters (the MCP editor sends a server's `url` as typed, so a key the
// user put in the URL would otherwise come back in clear). Both are redacted
// in place, the rest of the URL kept as written (no normalization), so the
// row still proves which endpoint was sent. The userinfo ends at the last
// `@` before the path, as the URL parser reads it, so an unencoded `@` in
// a password is part of the password, not the start of the host.
const URL_IN_TEXT = /\b[a-z][a-z0-9+.-]*:\/\/[^\s"'<>`]+/gi;
const USERINFO = /^([a-z][a-z0-9+.-]*:\/\/)([^/?#]*)@/i;

export function redactUrl(url) {
  let out = url.replace(USERINFO, (m, scheme, userinfo) => {
    const at = userinfo.indexOf(":");
    const parts =
      at < 0 ? [userinfo] : [userinfo.slice(0, at), userinfo.slice(at + 1)];
    return `${scheme}${parts.map(redactBodyValue).join(":")}@`;
  });
  const q = out.indexOf("?");
  if (q < 0) return out;
  const hash = out.indexOf("#", q);
  const query = hash < 0 ? out.slice(q + 1) : out.slice(q + 1, hash);
  const pairs = query.split("&").map((pair) => {
    const eq = pair.indexOf("=");
    if (eq < 0) return pair;
    const key = pair.slice(0, eq);
    let name = key;
    try {
      name = decodeURIComponent(key);
    } catch {
      // keep the raw key
    }
    const value = pair.slice(eq + 1);
    return `${key}=${SECRET_KEY.test(name) ? redactBodyValue(value) : redactUrlsIn(value)}`;
  });
  return `${out.slice(0, q + 1)}${pairs.join("&")}${hash < 0 ? "" : out.slice(hash)}`;
}

export function redactUrlsIn(text) {
  return text.replace(URL_IN_TEXT, redactUrl);
}

// `secret` is inherited by the whole subtree: a string in an array under
// `api_keys`, an object under `credentials`, every value of an `env` map.
function redactNode(node, secret = false) {
  if (typeof node === "string")
    return secret ? redactBodyValue(node) : redactUrlsIn(node);
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
      const params = new URLSearchParams();
      for (const [key, value] of new URLSearchParams(raw))
        params.append(
          key,
          all || SECRET_KEY.test(key) ? "<redacted>" : redactUrlsIn(value),
        );
      shown = params.toString();
    } else {
      return `<non-JSON body, ${raw.length} chars>`;
    }
  }
  return shown.length > limit ? `${shown.slice(0, limit)}…` : shown;
}
