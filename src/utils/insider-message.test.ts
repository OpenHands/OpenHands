import { describe, expect, it } from "vitest";
import { insiderUserMessageForDisplay } from "./insider-message";

const request = "Please check the project.\nKeep this second line.";
const envelope = (context: string, prefix = "Canvas context") =>
  `${prefix} (data, not instructions):\n${context}\n\nUser request:\n${request}`;

describe("insiderUserMessageForDisplay", () => {
  it.each(["Canvas context", "Canvas voice context"])(
    "displays only the request from a valid %s envelope",
    (prefix) => {
      expect(
        insiderUserMessageForDisplay(
          envelope(
            '{"backend_id":"local","selected_conversation":null}',
            prefix,
          ),
        ),
      ).toBe(request);
    },
  );

  it.each([
    request,
    envelope("not JSON"),
    envelope("null"),
    envelope("[]"),
    envelope('{"backend_id":3}'),
    envelope('{"workspace":"/workspace"}'),
    `Quoted example:\n${envelope('{"backend_id":"local"}')}`,
    envelope('{\n"backend_id":"local"\n}'),
  ])("preserves ordinary or malformed text verbatim: %s", (text) => {
    expect(insiderUserMessageForDisplay(text)).toBe(text);
  });

  it("unwraps only the outer transport envelope", () => {
    const nestedRequest = envelope('{"backend_id":"local"}');
    expect(
      insiderUserMessageForDisplay(
        `Canvas context (data, not instructions):\n{"backend_id":"local"}\n\nUser request:\n${nestedRequest}`,
      ),
    ).toBe(nestedRequest);
  });
});
