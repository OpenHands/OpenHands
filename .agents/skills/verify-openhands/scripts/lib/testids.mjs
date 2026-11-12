// Static check of the test ids the feature map drives against the source
// tree, so a renamed `data-testid` is noticed before any stack is launched.
//
// A cited id resolves when it appears as a string literal in src/, or when a
// prefix of it (cut at a hyphen) does: Canvas builds many ids from a prefix
// prop or a template (`${testIdPrefix}-name`, `plugin-card-${id}`), so the
// literal before the dynamic part is what the source holds. A one-word prefix
// counts only when the source really builds ids from it (`automation-${x}`):
// any id starting with a common word such as `automation` or `edit` would
// otherwise resolve, removed ids included. An id may also end in the literal
// tail of a template (`${type}-message`). Anything left is
// reported for a human or the live pass to judge; it is not proof of drift
// on its own (the id may be assembled in a way this heuristic misses).
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

/** Every `testid=...` handle cited in the map's feature files, with its families. */
export function citedTestids(mapDir, files) {
  const cited = new Map();
  for (const file of files) {
    const text = readFileSync(join(mapDir, file), "utf8");
    for (const m of text.matchAll(/testid=([A-Za-z0-9_./:$-]+)/g)) {
      const id = m[1].replace(/[.,;:)]+$/, "");
      if (!id) continue;
      if (!cited.has(id)) cited.set(id, new Set());
      cited.get(id).add(file);
    }
  }
  return cited;
}

function sourceFiles(dir, out = []) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name !== "node_modules") sourceFiles(path, out);
    } else if (
      /\.(tsx?|m?js)$/.test(entry.name) &&
      !/\.test\.[mc]?[jt]sx?$/.test(entry.name)
    ) {
      out.push(path);
    }
  }
  return out;
}

/**
 * String literals and template heads in the source tree that can be test ids.
 * The set also carries `heads` (template heads, `prefix-${...}`) and `tails`
 * (template tails, `${...}-suffix`).
 */
export function sourceLiterals(srcDir) {
  const literals = new Set();
  literals.heads = new Set();
  literals.tails = new Set();
  for (const file of sourceFiles(srcDir)) {
    const text = readFileSync(file, "utf8");
    for (const m of text.matchAll(/["'`]([a-z][a-z0-9_.:/-]*[a-z0-9])["'`$]/g))
      literals.add(m[1]);
    // `prefix-${dynamic}`: the head before the first placeholder.
    for (const m of text.matchAll(/`([a-z][a-z0-9_.:/-]*)-\$\{/g)) {
      literals.add(m[1]);
      literals.heads.add(m[1]);
    }
    // `${dynamic}-suffix`: the literal tail after the last placeholder.
    for (const m of text.matchAll(/\}-([a-z][a-z0-9_.:/-]*[a-z0-9])`/g))
      literals.tails.add(m[1]);
  }
  return literals;
}

/**
 * Which cited ids the source tree does not account for.
 * @returns {{ cited: number, unresolved: Array<{ id: string, files: string[] }> }}
 */
export function resolveTestids(cited, literals) {
  const heads = literals.heads ?? new Set();
  const tails = literals.tails ?? new Set();
  const unresolved = [];
  for (const [id, files] of cited) {
    if (literals.has(id)) continue;
    const parts = id.split("-");
    let found = false;
    for (let i = parts.length - 1; i >= 1 && !found; i -= 1) {
      const prefix = parts.slice(0, i).join("-");
      if (
        prefix.length >= 4 &&
        literals.has(prefix) &&
        (prefix.includes("-") || heads.has(prefix))
      )
        found = true;
    }
    for (let i = 1; i < parts.length && !found; i += 1) {
      const tail = parts.slice(i).join("-");
      if (tail.length >= 4 && tails.has(tail)) found = true;
    }
    if (!found) unresolved.push({ id, files: [...files].sort() });
  }
  return {
    cited: cited.size,
    unresolved: unresolved.sort((a, b) => a.id.localeCompare(b.id)),
  };
}

// ---------------------------------------------------------------------------
// The reverse check: controls the source renders that no recipe names.
//
// `map coverage` works per route and per component directory, so a page counts
// as mapped once any of it is. A button on a mapped page that no bullet ever
// operates is invisible to it, and that is where "the button does nothing"
// reports come from. This lists every interactive element in src/ that carries
// a literal test id the map never mentions. It is a list of candidates, not of
// defects: a recipe may reach the control by role or label instead, and an id
// may be decorative. Each one is either operated in a bullet, or left out on
// purpose.
// ---------------------------------------------------------------------------

const INTERACTIVE_TAG =
  /^(button|a|input|select|textarea|summary|Link|NavLink)$|(Button|Switch|Toggle|Checkbox|Radio|Input|Select|Dropdown|Combobox|Link|Tab|MenuItem|Option|Trigger)$/;
const INTERACTIVE_ATTR =
  /\b(onClick|onChange|onSubmit|onSelect|onToggle|onValueChange|onPress|onKeyDown)=|\brole="(button|menuitem|menuitemradio|menuitemcheckbox|tab|switch|checkbox|radio|option|combobox|link)"/;

/** The JSX opening tag around `index`: its name and text, or null. */
function openingTagAt(text, index) {
  // The nearest `<Name` before the attribute is not always its element: an
  // element passed as an earlier prop (`icon={<EditIcon />}`) sits between
  // them and has already closed. Walk back until a tag encloses the index.
  let start = index;
  while (index - start < 4000) {
    start = text.lastIndexOf("<", start - 1);
    if (start < 0) return null;
    if (!/[A-Za-z]/.test(text[start + 1] ?? "")) continue;
    // The tag ends at the first `>` outside a `{...}` expression.
    let depth = 0;
    let end = -1;
    for (let i = start; i < text.length && i < start + 4000; i += 1) {
      const ch = text[i];
      if (ch === "{") depth += 1;
      else if (ch === "}") depth -= 1;
      else if (ch === ">" && depth === 0) {
        end = i;
        break;
      }
    }
    if (end < index) continue;
    const tag = text.slice(start, end + 1);
    return { name: /^<([A-Za-z][A-Za-z0-9.]*)/.exec(tag)[1], tag };
  }
  return null;
}

/**
 * Elements in component source that carry a literal test id.
 * @returns {Array<{ id: string, file: string, tag: string, interactive: boolean }>}
 *   `file` is relative to `repoRoot`; one row per id and file.
 */
export function sourceControls(srcDir, repoRoot) {
  const rows = new Map();
  for (const file of sourceFiles(srcDir)) {
    if (!file.endsWith(".tsx")) continue;
    const rel = file
      .slice(repoRoot.length + 1)
      .split("\\")
      .join("/");
    if (/(^|\/)(mocks|__tests__|__mocks__|dev)\//.test(rel)) continue;
    const text = readFileSync(file, "utf8");
    for (const m of text.matchAll(
      /\b(?:data-testid|testId)=(?:"([a-z][a-z0-9_-]*)"|\{"([a-z][a-z0-9_-]*)"\})/g,
    )) {
      const id = m[1] ?? m[2];
      const found = openingTagAt(text, m.index);
      if (!found) continue;
      const interactive =
        INTERACTIVE_TAG.test(found.name) || INTERACTIVE_ATTR.test(found.tag);
      const key = `${id}\n${rel}`;
      const before = rows.get(key);
      if (!before || (interactive && !before.interactive))
        rows.set(key, { id, file: rel, tag: found.name, interactive });
    }
  }
  return [...rows.values()].sort(
    (a, b) => a.file.localeCompare(b.file) || a.id.localeCompare(b.id),
  );
}

/** Every id-shaped word in the map's text (recipes, prose and Gotchas alike). */
export function mapWords(mapText) {
  const words = new Set();
  for (const m of mapText.matchAll(/[a-z][a-z0-9_-]*[a-z0-9]/g))
    words.add(m[0]);
  return words;
}

/**
 * Split the source's controls into those the map names and those it does not.
 * @param {ReturnType<typeof sourceControls>} controls
 * @param {Set<string>} words from mapWords
 * @param {{ all?: boolean }} [options] all: count non-interactive elements too
 */
export function unnamedControls(controls, words, options = {}) {
  const considered = controls.filter((c) => options.all || c.interactive);
  const unnamed = considered.filter((c) => !words.has(c.id));
  return { considered: considered.length, unnamed };
}
