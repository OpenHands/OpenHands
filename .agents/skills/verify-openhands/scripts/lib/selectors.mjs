// ---------------------------------------------------------------------------
// Selector grammar shared with the CLI help:
//   segments joined by " >> "; each segment is one of
//   testid=ID | role=ROLE[name="Name"][exact][checked=true]... (also
//   [name^="Prefix"] and [name*="part"]) | text=Text |
//   text="Exact" | label=Label | placeholder=Text | title=Text | alt=Text |
//   nth=N | has-text=Text | visible | any Playwright/CSS selector.
// ---------------------------------------------------------------------------
export function unquote(value) {
  const trimmed = value.trim();
  if (
    (trimmed.startsWith('"') && trimmed.endsWith('"')) ||
    (trimmed.startsWith("'") && trimmed.endsWith("'"))
  ) {
    return { text: trimmed.slice(1, -1), quoted: true };
  }
  return { text: trimmed, quoted: false };
}

export function parseRole(spec) {
  const match = /^([a-z]+)((?:\[[^\]]*\])*)$/i.exec(spec.trim());
  if (!match) throw new Error(`Bad role segment: role=${spec}`);
  const options = {};
  for (const attr of match[2].matchAll(/\[([^\]=]+)(?:=([^\]]*))?\]/g)) {
    const key = attr[1].trim();
    const raw = attr[2];
    const escape = (t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    if (key === "name^" || key === "name*") {
      // [name^="Start"] prefix match, [name*="part"] case-insensitive contains.
      const { text } = unquote(raw ?? "");
      options.name =
        key === "name^"
          ? new RegExp(`^${escape(text)}`)
          : new RegExp(escape(text), "i");
    } else if (key === "name") {
      const { text } = unquote(raw ?? "");
      if (text.startsWith("/") && text.lastIndexOf("/") > 0) {
        const end = text.lastIndexOf("/");
        options.name = new RegExp(text.slice(1, end), text.slice(end + 1));
      } else {
        options.name = text;
      }
    } else if (key === "level") {
      options.level = Number(raw);
    } else if (
      [
        "exact",
        "checked",
        "disabled",
        "expanded",
        "pressed",
        "selected",
        "includeHidden",
      ].includes(key)
    ) {
      options[key] = raw === undefined ? true : raw === "true";
    } else {
      throw new Error(`Unsupported role attribute [${key}]`);
    }
  }
  return { role: match[1], options };
}

export function applySegment(scope, segment) {
  const seg = segment.trim();
  const eq = seg.indexOf("=");
  const engine = eq > 0 ? seg.slice(0, eq) : seg;
  const value = eq > 0 ? seg.slice(eq + 1) : "";
  switch (engine) {
    case "testid": {
      // testid=name[data-state="open"] narrows by extra attributes via CSS.
      const bracket = value.indexOf("[");
      if (bracket > 0) {
        const id = unquote(value.slice(0, bracket)).text;
        return scope.locator(`[data-testid="${id}"]${value.slice(bracket)}`);
      }
      return scope.getByTestId(unquote(value).text);
    }
    case "role": {
      const { role, options } = parseRole(value);
      return scope.getByRole(role, options);
    }
    case "text": {
      const { text, quoted } = unquote(value);
      return scope.getByText(text, { exact: quoted });
    }
    case "label": {
      const { text, quoted } = unquote(value);
      return scope.getByLabel(text, { exact: quoted });
    }
    case "placeholder": {
      const { text, quoted } = unquote(value);
      return scope.getByPlaceholder(text, { exact: quoted });
    }
    case "title": {
      const { text, quoted } = unquote(value);
      return scope.getByTitle(text, { exact: quoted });
    }
    case "alt": {
      const { text, quoted } = unquote(value);
      return scope.getByAltText(text, { exact: quoted });
    }
    case "nth":
      return scope.nth(Number(value));
    case "has-text":
    case "visible":
      if (typeof scope.filter !== "function")
        throw new Error(
          `${engine}= narrows the previous segment; start with a selector, e.g. 'role=row >> ${seg}' (or use text= on its own)`,
        );
      return engine === "has-text"
        ? scope.filter({ hasText: unquote(value).text })
        : scope.filter({ visible: true });
    default:
      return scope.locator(seg);
  }
}

export function buildLocator(root, selector) {
  if (!selector) throw new Error("A selector is required");
  return selector
    .split(/\s+>>\s+/)
    .reduce((scope, segment) => applySegment(scope, segment), root);
}

// Plain CSS for selectors made only of testid= and CSS segments (used by the
// in-page MutationObserver of click --observe); null when Playwright engines
// (role=, text=, nth= ...) are needed.
export function toCss(selector) {
  const parts = [];
  for (const raw of String(selector).split(" >> ")) {
    const seg = raw.trim();
    if (seg.startsWith("testid=")) {
      const value = seg.slice("testid=".length);
      const bracket = value.indexOf("[");
      const id = unquote(bracket > 0 ? value.slice(0, bracket) : value).text;
      parts.push(
        `[data-testid="${id.replace(/"/g, '\\"')}"]${bracket > 0 ? value.slice(bracket) : ""}`,
      );
    } else if (/^[a-z-]+=/.test(seg) || seg === "visible") {
      return null;
    } else {
      parts.push(seg);
    }
  }
  return parts.join(" ");
}
