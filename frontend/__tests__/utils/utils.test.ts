import { describe, it, expect, vi, test } from "vitest";
import {
  formatTimestamp,
  getExtension,
  removeApiKey,
} from "../../src/utils/utils";
import { getStatusText } from "#/utils/utils";
import { AgentState } from "#/types/agent-state";
import { I18nKey } from "#/i18n/declaration";

// Mock translations
const t = (key: string) => {
  const translations: { [key: string]: string } = {
    COMMON$WAITING_FOR_SANDBOX: "Waiting For Sandbox",
    COMMON$STOPPING: "Stopping",
    COMMON$STARTING: "Starting",
    COMMON$SERVER_STOPPED: "Server stopped",
    COMMON$RUNNING: "Running",
    CONVERSATION$READY: "Ready",
    CONVERSATION$ERROR_STARTING_CONVERSATION: "Error starting conversation",
  };
  return translations[key] || key;
};

test("removeApiKey", () => {
  const data = [{ args: { LLM_API_KEY: "key", LANGUAGE: "en" } }];
  expect(removeApiKey(data)).toEqual([{ args: { LANGUAGE: "en" } }]);
});

test("getExtension", () => {
  expect(getExtension("main.go")).toBe("go");
  expect(getExtension("get-extension.test.ts")).toBe("ts");
  expect(getExtension("directory")).toBe("");
});

test("formatTimestamp", () => {
  const morningDate = new Date("2021-10-10T10:10:10.000").toISOString();
  expect(formatTimestamp(morningDate)).toBe("10/10/2021, 10:10:10");

  const eveningDate = new Date("2021-10-10T22:10:10.000").toISOString();
  expect(formatTimestamp(eveningDate)).toBe("10/10/2021, 22:10:10");
});

describe("getStatusText", () => {
  it("returns STOPPING when pausing", () => {
    const result = getStatusText({
      isPausing: true,
      isTask: false,
      taskStatus: null,
      taskDetail: null,
      isStartingStatus: false,
      isStopStatus: false,
      curAgentState: AgentState.RUNNING,
      t,
    });

    expect(result).toBe(t(I18nKey.COMMON$STOPPING));
  });

  it("formats task status when polling a task", () => {
    const result = getStatusText({
      isPausing: false,
      isTask: true,
      taskStatus: "WAITING_FOR_SANDBOX",
      taskDetail: null,
      isStartingStatus: false,
      isStopStatus: false,
      curAgentState: AgentState.RUNNING,
      t,
    });

    expect(result).toBe(t(I18nKey.COMMON$WAITING_FOR_SANDBOX));
  });

  it("returns task detail when task status is ERROR and detail exists", () => {
    const result = getStatusText({
      isPausing: false,
      isTask: true,
      taskStatus: "ERROR",
      taskDetail: "Sandbox failed",
      isStartingStatus: false,
      isStopStatus: false,
      curAgentState: AgentState.RUNNING,
      t,
    });

    expect(result).toBe("Sandbox failed");
  });

  it("returns translated error when task status is ERROR and no detail", () => {
    const result = getStatusText({
      isPausing: false,
      isTask: true,
      taskStatus: "ERROR",
      taskDetail: null,
      isStartingStatus: false,
      isStopStatus: false,
      curAgentState: AgentState.RUNNING,
      t,
    });

    expect(result).toBe(
     t(I18nKey.CONVERSATION$ERROR_STARTING_CONVERSATION),
    );
  });

  it("returns READY translation when task is ready", () => {
    const result = getStatusText({
      isPausing: false,
      isTask: true,
      taskStatus: "READY",
      taskDetail: null,
      isStartingStatus: false,
      isStopStatus: false,
      curAgentState: AgentState.RUNNING,
      t,
    });

    expect(result).toBe(t(I18nKey.CONVERSATION$READY));
  });

  it("returns STARTING when starting status is true", () => {
    const result = getStatusText({
      isPausing: false,
      isTask: false,
      taskStatus: null,
      taskDetail: null,
      isStartingStatus: true,
      isStopStatus: false,
      curAgentState: AgentState.INIT,
      t,
    });

    expect(result).toBe(t(I18nKey.COMMON$STARTING));
  });

  it("returns SERVER_STOPPED when stop status is true", () => {
    const result = getStatusText({
      isPausing: false,
      isTask: false,
      taskStatus: null,
      taskDetail: null,
      isStartingStatus: false,
      isStopStatus: true,
      curAgentState: AgentState.STOPPED,
      t,
    });

    expect(result).toBe(t(I18nKey.COMMON$SERVER_STOPPED));
  });

  it("returns errorMessage when agent state is ERROR", () => {
    const result = getStatusText({
      isPausing: false,
      isTask: false,
      taskStatus: null,
      taskDetail: null,
      isStartingStatus: false,
      isStopStatus: false,
      curAgentState: AgentState.ERROR,
      errorMessage: "Something broke",
      t,
    });

    expect(result).toBe("Something broke");
  });

  it("returns default RUNNING status", () => {
    const result = getStatusText({
      isPausing: false,
      isTask: false,
      taskStatus: null,
      taskDetail: null,
      isStartingStatus: false,
      isStopStatus: false,
      curAgentState: AgentState.RUNNING,
      t,
    });

    expect(result).toBe(t(I18nKey.COMMON$RUNNING));
  });
});
