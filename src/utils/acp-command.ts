// ``shell-quote`` is a CJS module that does ``module.exports = { parse, quote }``;
// Vite's ESM interop can resolve a default/namespace import but not named
// imports against that shape (the dev server crashes with "does not provide
// an export named 'parse'"). Namespace import works on both the dev server
// and the Rollup-based prod build. Used only for the ``quote`` direction —
// see ``parseCommand`` below for why we don't use ``shell-quote.parse``.
import * as shellQuote from "shell-quote";

const { quote } = shellQuote;

/**
 * Parse a single-string command into argv tokens for ``acp_command``.
 *
 * Used by the Settings → Agent textarea — the user types one human-readable
 * command (e.g. ``bash -c "echo hello world"``) and we convert it into the
 * ``string[]`` shape that the agent-server's ``ACPAgent.acp_command``
 * expects. The agent-server passes that array straight to
 * ``subprocess.create_subprocess_exec``; no shell is involved on the spawn
 * side, so this parser only needs to handle argv-style word splitting
 * with quote/escape support — *not* shell metasyntax.
 *
 * Why a custom tokenizer and **not** ``shell-quote.parse``:
 *
 * ``shell-quote.parse`` treats ``?``, ``*``, ``$VAR``, redirects, and
 * comments as shell syntax and emits non-string AST nodes for them.
 * Filtering to strings would silently drop entire argv tokens. The
 * concrete data-corruption case is a URL with a query string —
 *
 *     node acp.js --endpoint https://example.com/acp?tenant=abc
 *
 * ``shell-quote`` reads ``?tenant=abc`` as a glob pattern and returns
 * ``["node","acp.js","--endpoint",{op:"glob",…}]``, so the saved
 * ``acp_command`` becomes ``["node","acp.js","--endpoint"]`` — the URL
 * vanishes. The agent-server then spawns a broken command and the user
 * gets a confusing runtime error far from the configuration UI.
 *
 * The replacement tokenizer treats every non-whitespace, non-quote
 * character as part of the current token: ``?``, ``*``, ``$``, ``|``,
 * ``>``, ``#``, backticks all round-trip verbatim. Shell-only constructs
 * (pipes, redirects, env-var expansion, command substitution) would
 * land as literal argv entries — which is what the user typed and what
 * ``subprocess.create_subprocess_exec`` will see. That's correct: a
 * user who types ``foo | bar`` into the Settings → Agent textarea is
 * configuring a literal command, not a shell pipeline; ``foo`` doesn't
 * actually pipe into ``bar``, but neither does it silently disappear.
 *
 * Quoting rules supported:
 *   - whitespace separates tokens
 *   - single quotes: literal until the next ``'`` (no escapes inside,
 *     matching POSIX shell)
 *   - double quotes: literal until the next ``"`` (with ``\\"`` and
 *     ``\\\\`` honored as escapes; no $-expansion)
 *   - backslash outside quotes: escapes the next character (whitespace,
 *     quote, or anything else — turns it into a literal)
 *   - explicit empty quoted segments (``""`` / ``''``) produce an
 *     empty-string token, matching the round-trip rule in
 *     ``formatCommand``
 *
 * Unterminated quotes are tolerated: the current token closes at EOF
 * with whatever was accumulated. A throw here would crash the
 * Settings → Agent page mid-render; the Save button is already gated
 * on a non-empty argv so a recoverable miss can't be silently saved
 * either way.
 */
export function parseCommand(value: string): string[] {
  const tokens: string[] = [];
  let i = 0;
  let current = "";
  let inToken = false;

  while (i < value.length) {
    const ch = value[i];

    if (ch === " " || ch === "\t" || ch === "\n" || ch === "\r") {
      if (inToken) {
        tokens.push(current);
        current = "";
        inToken = false;
      }
      i += 1;
      continue;
    }

    if (ch === "'") {
      // Single-quoted segment: literal until the next single quote.
      // No escapes inside (POSIX shell semantics).
      inToken = true;
      i += 1;
      while (i < value.length && value[i] !== "'") {
        current += value[i];
        i += 1;
      }
      // Skip the closing quote if present. Unterminated → EOF closes.
      if (i < value.length) i += 1;
      continue;
    }

    if (ch === '"') {
      // Double-quoted segment: literal with backslash escapes for
      // ``\\"`` and ``\\\\``. We intentionally do NOT expand $VAR
      // (so a user typing ``"--key=$X"`` keeps it literal — the
      // agent-server doesn't run a shell anyway). Other backslash
      // sequences pass through verbatim (matches what most users
      // expect when copying paths with backslashes from Windows
      // examples; corner-case differences from POSIX aren't worth
      // the complexity here).
      inToken = true;
      i += 1;
      while (i < value.length && value[i] !== '"') {
        if (value[i] === "\\" && i + 1 < value.length) {
          const next = value[i + 1];
          if (next === '"' || next === "\\") {
            current += next;
            i += 2;
            continue;
          }
        }
        current += value[i];
        i += 1;
      }
      if (i < value.length) i += 1;
      continue;
    }

    if (ch === "\\" && i + 1 < value.length) {
      // Unquoted backslash escapes the next character (whitespace,
      // quote, glob char, anything). Useful for typing literal spaces
      // in a path or escaping a literal quote.
      inToken = true;
      current += value[i + 1];
      i += 2;
      continue;
    }

    // Every other character — including ``?``, ``*``, ``$``, ``|``,
    // ``>``, ``#``, ``&``, ``;``, ``(``, ``)`` — is a literal part of
    // the current token. Shell-metasyntax filtering happens at the
    // OS/shell boundary, which we don't cross.
    inToken = true;
    current += ch;
    i += 1;
  }

  if (inToken) {
    tokens.push(current);
  }
  return tokens;
}

// Tokens that need shell-quoting when rendering back to a string —
// whitespace, quotes, backslashes, redirects/pipes/globs, and the
// command-separators. ``@``, ``/``, ``-``, ``.``, ``+``, ``=`` and
// other punctuation that's common in package names and URLs are
// safe in argv-only contexts (the agent-server execs the array, no
// shell intermediary), so we leave them alone — otherwise
// ``npx -y @org/pkg`` would render as ``npx -y \@org/pkg`` and that's
// a hostile read-back in the textarea.
//
// Note: ``?``, ``*``, ``$``, ``|``, ``>``, ``#``, ``&``, ``;``, ``(``,
// ``)`` no longer carry shell meaning in {@link parseCommand} (which
// is purely an argv tokenizer), so a token containing them would
// round-trip fine without quoting. We still quote on output because
// users frequently switch between this textarea and an actual shell
// (copy-paste workflows), and quoting matches the conservative
// expectation "if it would need quoting in a shell, show it quoted."
const SHELL_UNSAFE = /[\s"'\\$`&|;<>(){}*?#!~[\]]/;

/**
 * Render a ``string[]`` argv back into a single string the textarea
 * can display. Tokens that *would* need shell quoting (whitespace,
 * quotes, redirects, …) go through ``shell-quote.quote`` for correct
 * escaping; tokens that are already shell-safe (the overwhelming
 * majority of package names and CLI flags) round-trip verbatim. The
 * output remains a valid input to {@link parseCommand}.
 */
export function formatCommand(command: readonly string[]): string {
  return command
    .map((tok) =>
      // Quote any token that:
      //   * contains a shell-significant character (whitespace, quotes,
      //     redirects, …), so it doesn't get re-split on parse, OR
      //   * is the empty string — without explicit quoting,
      //     ``["bash", "-c", ""]`` would render as ``"bash -c "`` and
      //     round-trip back to ``["bash", "-c"]``, silently dropping
      //     the (rare but valid) empty argument.
      SHELL_UNSAFE.test(tok) || tok === "" ? quote([tok]) : tok,
    )
    .join(" ");
}
