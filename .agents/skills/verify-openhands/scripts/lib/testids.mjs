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
