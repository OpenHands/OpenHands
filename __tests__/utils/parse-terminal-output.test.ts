import { describe, it, expect } from "vitest";
import { parseTerminalOutput } from "../../src/utils/parse-terminal-output";

describe("parseTerminalOutput", () => {
  it("should parse the command, env, and symbol", () => {
    const raw =
      "web_scraper.py\r\n\r\n[Python Interpreter: /openhands/poetry/openhands-5O4_aCHf-py3.11/bin/python]\nopenhands@659478cb008c:/workspace $ ";

    const parsed = parseTerminalOutput(raw);
    expect(parsed).toBe("web_scraper.py");
  });

  it("should parse even if there is no output", () => {
    const raw =
      "[Python Interpreter: /openhands/poetry/openhands-5O4_aCHf-py3.11/bin/python]\nopenhands@659478cb008c:/workspace $ ";

    const parsed = parseTerminalOutput(raw);
    expect(parsed).toBe("");
  });

  it("should return the string if it doesn't match the regex", () => {
    const raw = "web_scraper.py";
    const parsed = parseTerminalOutput(raw);
    expect(parsed).toBe("web_scraper.py");
  });
});
