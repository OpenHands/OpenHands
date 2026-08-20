import { describe, expect, it } from "vitest";
import { formatCommand, parseCommand } from "#/utils/acp-command";

describe("parseCommand", () => {
  it("splits a simple npx invocation into argv tokens", () => {
    expect(
      parseCommand("npx -y @agentclientprotocol/claude-agent-acp"),
    ).toEqual(["npx", "-y", "@agentclientprotocol/claude-agent-acp"]);
  });

  it("respects double-quoted segments — the headline regression .split fix", () => {
    // The old `.split(/\s+/)` implementation turned this into
    // ``["bash", "-c", "\"echo", "hello", "world\""]`` and the spawn
    // would either misbehave or fail in a confusing place. The
    // quote-aware tokenizer keeps the quoted segment intact.
    expect(parseCommand('bash -c "echo hello world"')).toEqual([
      "bash",
      "-c",
      "echo hello world",
    ]);
  });

  it("respects single-quoted segments and embedded whitespace", () => {
    expect(parseCommand("env FOO='bar baz' npx -y my-acp")).toEqual([
      "env",
      "FOO=bar baz",
      "npx",
      "-y",
      "my-acp",
    ]);
  });

  it("preserves URLs with query strings — the headline shell-quote-glob fix", () => {
    // Regression guard for the silent-corruption bug:
    //
    //   node acp.js --endpoint https://example.com/acp?tenant=abc
    //
    // ``shell-quote.parse`` used to read ``?tenant=abc`` as a glob
    // pattern and drop the entire URL token, so the saved
    // ``acp_command`` became ``["node", "acp.js", "--endpoint"]``.
    // The spawn would then fail with a confusing "missing endpoint"
    // error far from the Settings → Agent page that caused it.
    //
    // The custom tokenizer treats ``?`` as a literal — same for
    // every other shell metacharacter. The URL round-trips intact.
    expect(
      parseCommand("node acp.js --endpoint https://example.com/acp?tenant=abc"),
    ).toEqual([
      "node",
      "acp.js",
      "--endpoint",
      "https://example.com/acp?tenant=abc",
    ]);
  });

  it("preserves URLs with multiple query params", () => {
    // ``&`` is also literal — same reason.
    expect(parseCommand("curl https://x.com?a=1&b=2")).toEqual([
      "curl",
      "https://x.com?a=1&b=2",
    ]);
  });

  it("preserves shell metacharacters as literal argv tokens", () => {
    // Pipes, redirects, semicolons, glob chars, ``$``, backticks,
    // ``#`` all round-trip as literal characters within the surrounding
    // token. The agent-server uses ``subprocess.create_subprocess_exec``
    // (no shell intermediary), so a user typing ``foo | bar`` is
    // configuring two literal argv entries — not a shell pipeline.
    // The user's helper text steers them to ``bash -c '…'`` if they
    // actually want shell features.
    expect(parseCommand("foo | bar")).toEqual(["foo", "|", "bar"]);
    expect(parseCommand("foo > log.txt")).toEqual(["foo", ">", "log.txt"]);
    expect(parseCommand("foo *.txt")).toEqual(["foo", "*.txt"]);
    expect(parseCommand("foo $X")).toEqual(["foo", "$X"]);
    expect(parseCommand("foo `bar`")).toEqual(["foo", "`bar`"]);
    expect(parseCommand("foo # comment")).toEqual(["foo", "#", "comment"]);
    expect(parseCommand("foo && bar")).toEqual(["foo", "&&", "bar"]);
    expect(parseCommand("foo; bar")).toEqual(["foo;", "bar"]);
  });

  it("treats blank input as an empty argv", () => {
    expect(parseCommand("")).toEqual([]);
    expect(parseCommand("   \t\n   ")).toEqual([]);
  });

  it("honors backslash escapes outside quotes", () => {
    // ``foo\ bar`` is one token containing a literal space — the same
    // contract POSIX shells provide. Lets the user type paths with
    // spaces without reaching for quotes.
    expect(parseCommand("foo\\ bar")).toEqual(["foo bar"]);
    // An escaped quote becomes a literal quote in the token.
    expect(parseCommand('foo\\"bar')).toEqual(['foo"bar']);
  });

  it('honors ``\\\\`` and ``\\"`` inside double-quoted segments', () => {
    expect(parseCommand('bash -c "echo \\"hi\\""')).toEqual([
      "bash",
      "-c",
      'echo "hi"',
    ]);
    expect(parseCommand('"foo\\\\bar"')).toEqual(["foo\\bar"]);
  });

  it("does not env-expand $VAR refs — keeps them as literal", () => {
    // The forbidden outcome would be the tokenizer reading
    // ``process.env.ANTHROPIC_API_KEY`` and inlining its value into
    // the persisted ``acp_command`` — that would leak a host env var
    // into settings on every save. The tokenizer reads ``$NAME`` as
    // a literal substring of the token, so the user's typed text
    // survives verbatim. Provider credentials belong in the Secrets
    // panel (request.secrets), never inlined into the command.
    const result = parseCommand("npx $ANTHROPIC_API_KEY");
    expect(result).toEqual(["npx", "$ANTHROPIC_API_KEY"]);
    // Pin the no-leak contract: no ``sk-…`` token sneaks through
    // from the host env (which is also unset here, but still).
    expect(result.some((t) => /sk-ant-/.test(t))).toBe(false);
  });

  it("does not run subshells: $(…) and backticks become literal tokens", () => {
    // The forbidden outcome would be executing ``date`` and inlining
    // today's timestamp into the persisted command. The tokenizer
    // never invokes anything; both forms round-trip verbatim.
    expect(parseCommand("echo $(date)")).toEqual(["echo", "$(date)"]);
    expect(parseCommand("echo `date`")).toEqual(["echo", "`date`"]);
  });

  it("survives unterminated quotes without throwing", () => {
    // EOF closes the open quote; the partially-built token gets
    // pushed. A throw here would crash the Settings → Agent page
    // mid-render. The Save button gates on a non-empty argv anyway,
    // so a recoverable miss can't be silently persisted.
    expect(parseCommand('bash -c "unterminated')).toEqual([
      "bash",
      "-c",
      "unterminated",
    ]);
    expect(parseCommand("foo 'unterminated single")).toEqual([
      "foo",
      "unterminated single",
    ]);
  });
});

describe("formatCommand", () => {
  it("renders package-style tokens verbatim, no escaping of @ or /", () => {
    // The textarea is the only consumer of formatCommand. Escaping the
    // ``@`` in ``@org/pkg`` produces a hostile read-back (the user
    // copies their existing command, the textarea now shows
    // ``\@org/pkg``, they think we corrupted it). The agent-server
    // execs argv directly so the escape isn't load-bearing for
    // behaviour — only for display.
    expect(
      formatCommand(["npx", "-y", "@agentclientprotocol/claude-agent-acp"]),
    ).toBe("npx -y @agentclientprotocol/claude-agent-acp");
  });

  it("shell-quotes tokens that contain whitespace", () => {
    expect(formatCommand(["bash", "-c", "echo hello world"])).toBe(
      "bash -c 'echo hello world'",
    );
  });

  it("round-trips arbitrary argv arrays through parseCommand", () => {
    const cases: string[][] = [
      ["npx", "-y", "@agentclientprotocol/claude-agent-acp"],
      ["npx", "-y", "@zed-industries/codex-acp"],
      ["npx", "-y", "@google/gemini-cli", "--acp"],
      ["bash", "-c", "echo hello world"],
      ["env", "FOO=bar baz", "npx", "-y", "my-acp"],
      ["./bin/my-agent", "--flag=value"],
      // URL with query string — the headline silent-corruption case.
      ["node", "acp.js", "--endpoint", "https://example.com/acp?tenant=abc"],
      // URL with multiple params.
      ["curl", "https://x.com?a=1&b=2"],
      // Empty-string tokens are rare but valid (some CLIs treat an
      // empty positional as "no argument supplied" rather than missing).
      // Without explicit quoting in formatCommand they round-trip back
      // as fewer tokens, silently dropping the empty slot.
      ["bash", "-c", ""],
      ["program", "", "--flag"],
    ];
    for (const argv of cases) {
      expect(parseCommand(formatCommand(argv))).toEqual(argv);
    }
  });

  it("renders an empty argv as an empty string", () => {
    expect(formatCommand([])).toBe("");
  });

  it("explicitly quotes empty-string tokens so they survive the round trip", () => {
    // Direct assertion on the rendered form — without this rule,
    // formatCommand(["bash","-c",""]) would render ``"bash -c "`` and
    // parseCommand would return ``["bash", "-c"]``, losing the empty arg.
    expect(formatCommand(["bash", "-c", ""])).toBe("bash -c ''");
  });
});
