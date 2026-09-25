import { describe, expect, it } from "vitest";

import { evaluateBashObservation } from "../../tests/e2e/mock-llm/utils/bash-observation-oracle";

const COMMAND = "printf 'MOCK_TOKEN\\n'";
const TOKEN = "MOCK_TOKEN";
const EXPECTED = { command: COMMAND, token: TOKEN };

function terminalObservation(
  overrides: {
    command?: string | null;
    exitCode?: number | null;
    texts?: string[];
    isError?: boolean;
    timeout?: boolean;
    kind?: string;
  } = {},
) {
  return {
    kind: "ObservationEvent",
    source: "environment",
    tool_name: "terminal",
    observation: {
      kind: overrides.kind ?? "TerminalObservation",
      command: overrides.command === undefined ? COMMAND : overrides.command,
      exit_code: overrides.exitCode === undefined ? 0 : overrides.exitCode,
      is_error: overrides.isError ?? false,
      timeout: overrides.timeout ?? false,
      content: (overrides.texts ?? [`${TOKEN}\n`]).map((text) => ({
        type: "text",
        text,
      })),
    },
  };
}

describe("evaluateBashObservation", () => {
  it("passes when the exact command exits zero with the token in its output", () => {
    const verdict = evaluateBashObservation([terminalObservation()], EXPECTED);

    expect(verdict.status).toBe("success");
  });

  it("accepts the legacy ExecuteBashObservation kind", () => {
    const verdict = evaluateBashObservation(
      [terminalObservation({ kind: "ExecuteBashObservation" })],
      EXPECTED,
    );

    expect(verdict.status).toBe("success");
  });

  it("joins output split across content chunks before looking for the token", () => {
    const verdict = evaluateBashObservation(
      [terminalObservation({ texts: ["MOCK_", "TOKEN\n"] })],
      EXPECTED,
    );

    expect(verdict.status).toBe("success");
  });

  it("ignores an unrelated command even when it succeeds and prints the token", () => {
    const unrelated = terminalObservation({ command: `echo ${TOKEN}` });

    const verdict = evaluateBashObservation([unrelated], EXPECTED);

    expect(verdict.status).toBe("pending");
  });

  it("ignores bash-stream events that the old oracle accepted", () => {
    const staleBashStream = [
      { kind: "BashOutput", exit_code: 0, stdout: `${TOKEN}\n` },
      { kind: "BashOutput", exit_code: 0, stdout: null },
    ];

    const verdict = evaluateBashObservation(staleBashStream, EXPECTED);

    expect(verdict.status).toBe("pending");
  });

  it("fails an exit-zero match with empty output even if another event has the token", () => {
    const events = [
      terminalObservation({ texts: [] }),
      terminalObservation({ command: `echo ${TOKEN}` }),
    ];

    const verdict = evaluateBashObservation(events, EXPECTED);

    expect(verdict.status).toBe("failed");
    expect(verdict.detail).toMatch(/output did not contain/);
  });

  it("fails when the exact command exits nonzero", () => {
    const verdict = evaluateBashObservation(
      [terminalObservation({ exitCode: 1 })],
      EXPECTED,
    );

    expect(verdict.status).toBe("failed");
    expect(verdict.detail).toMatch(/exit code 1/);
  });

  it("fails when the observation is flagged as an error", () => {
    const verdict = evaluateBashObservation(
      [terminalObservation({ isError: true })],
      EXPECTED,
    );

    expect(verdict.status).toBe("failed");
  });

  it("fails when the command timed out", () => {
    const verdict = evaluateBashObservation(
      [terminalObservation({ timeout: true })],
      EXPECTED,
    );

    expect(verdict.status).toBe("failed");
  });

  it("keeps waiting while the command is still running", () => {
    const verdict = evaluateBashObservation(
      [terminalObservation({ exitCode: -1, texts: [] })],
      EXPECTED,
    );

    expect(verdict.status).toBe("pending");
  });

  it("prefers a later successful run of the same command over an earlier failure", () => {
    const events = [
      terminalObservation({ exitCode: 1 }),
      terminalObservation(),
    ];

    const verdict = evaluateBashObservation(events, EXPECTED);

    expect(verdict.status).toBe("success");
  });

  it("ignores malformed events", () => {
    const events = [null, "text", 42, {}, { observation: null }];

    const verdict = evaluateBashObservation(events, EXPECTED);

    expect(verdict.status).toBe("pending");
    expect(verdict.detail).toMatch(/5 events/);
  });
});
