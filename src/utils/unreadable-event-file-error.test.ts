import { HttpError } from "@openhands/typescript-client";
import { describe, expect, it } from "vitest";
import { getUnreadableEventFileReason } from "./unreadable-event-file-error";

// Built the way the shared client builds it: the parsed body is `response`.
const agentServerError = (status: number, exception: string) => {
  const body = { detail: "Internal Server Error", exception, error_id: "e1" };
  return new HttpError(
    status,
    "Internal Server Error",
    body,
    `HTTP request failed (${status} Internal Server Error): ${JSON.stringify(body)}`,
  );
};

const MISSING =
  "Missing event file: events/event-00552-15c60853-a045-4acf-915f-163fb47e6858.json";
const CUT_SHORT =
  '1 validation error for Event\n  Invalid JSON: EOF while parsing a string at line 1 column 172 [type=json_invalid, input_value=\'{"id":"74b8\', input_type=str]';

describe("getUnreadableEventFileReason", () => {
  it.each([
    ["an empty event file", MISSING],
    ["an event file cut short", CUT_SHORT],
  ])("returns the server's reason for %s", (_, exception) => {
    expect(getUnreadableEventFileReason(agentServerError(500, exception))).toBe(
      exception,
    );
  });

  it.each([
    ["another server failure", agentServerError(500, "disk is full")],
    ["another status", agentServerError(502, MISSING)],
    [
      "an event that fails validation for another reason",
      agentServerError(
        500,
        "1 validation error for Event\n  kind\n    Input tag 'X' found",
      ),
    ],
    ["a transport error that only echoes the text", new Error(MISSING)],
    [
      "an HTTP error without a parsed body",
      new HttpError(500, "Internal Server Error", MISSING),
    ],
    ["a non-error", undefined],
  ])("returns null for %s", (_, error) => {
    expect(getUnreadableEventFileReason(error)).toBeNull();
  });
});
