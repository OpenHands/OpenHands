import { describe, it, expect } from "vitest";
import { getStatusCode, getTaskStatusI18nKey } from "#/utils/status";
import { I18nKey } from "#/i18n/declaration";
import { ExecutionStatus } from "#/types/agent-server/core";
import type { AppConversationStartTaskStatus } from "#/api/conversation-service/agent-server-conversation-service.types";

describe("getStatusCode", () => {
  it("returns RUNNING_TASK when execution status is running", () => {
    const result = getStatusCode("OPEN", ExecutionStatus.RUNNING);
    expect(result).toBe(I18nKey.AGENT_STATUS$RUNNING_TASK);
  });

  it("returns WAITING_FOR_TASK when execution status is idle", () => {
    const result = getStatusCode("OPEN", ExecutionStatus.IDLE);
    expect(result).toBe(I18nKey.AGENT_STATUS$WAITING_FOR_TASK);
  });

  it("returns STOPPED when execution status is paused", () => {
    const result = getStatusCode("OPEN", ExecutionStatus.PAUSED);
    expect(result).toBe(I18nKey.CHAT_INTERFACE$STOPPED);
  });

  it("returns starting i18n key when task is running setup", () => {
    const result = getStatusCode("OPEN", null, "STARTING_CONVERSATION");
    expect(result).toBe(I18nKey.CONVERSATION$STARTING_CONVERSATION);
  });

  it("prioritises task ERROR over websocket CONNECTING", () => {
    const result = getStatusCode("CONNECTING", null, "ERROR");
    expect(result).toBe(I18nKey.AGENT_STATUS$ERROR_OCCURRED);
  });

  it("returns DISCONNECTED when websocket is closed and no task", () => {
    const result = getStatusCode("CLOSED", ExecutionStatus.IDLE);
    expect(result).toBe(I18nKey.CHAT_INTERFACE$DISCONNECTED);
  });

  it("returns COMMON$WAITING_FOR_SANDBOX when task is waiting for sandbox", () => {
    const result = getStatusCode("OPEN", null, "WAITING_FOR_SANDBOX");
    expect(result).toBe(I18nKey.COMMON$WAITING_FOR_SANDBOX);
  });

  it("falls back to starting i18n key for an unknown task status instead of throwing", () => {
    const unknownStatus =
      "FUTURE_STATUS_FROM_CLOUD" as AppConversationStartTaskStatus;
    const result = getStatusCode("OPEN", null, unknownStatus);
    expect(result).toBe(I18nKey.CONVERSATION$STARTING_CONVERSATION);
  });
});

describe("getTaskStatusI18nKey", () => {
  // Exhaustive coverage of the shared mapper over every AppConversationStartTaskStatus
  // member: the dedicated keys (WAITING_FOR_SANDBOX + the two setup keys), the
  // terminal READY/ERROR states (which now resolve to their own localized keys
  // instead of silently falling back to STARTING_CONVERSATION), and the group
  // that collapses to the generic "Starting" label (WORKING/PREPARING_REPOSITORY/
  // RUNNING_SETUP_SCRIPT). `as const` keeps the inputs typed as the union.
  it.each([
    ["WAITING_FOR_SANDBOX", I18nKey.COMMON$WAITING_FOR_SANDBOX],
    ["SETTING_UP_GIT_HOOKS", I18nKey.STATUS$SETTING_UP_GIT_HOOKS],
    ["SETTING_UP_SKILLS", I18nKey.STATUS$SETTING_UP_SKILLS],
    ["READY", I18nKey.CONVERSATION$READY],
    ["ERROR", I18nKey.COMMON$ERROR],
    ["WORKING", I18nKey.CONVERSATION$STARTING_CONVERSATION],
    ["PREPARING_REPOSITORY", I18nKey.CONVERSATION$STARTING_CONVERSATION],
    ["RUNNING_SETUP_SCRIPT", I18nKey.CONVERSATION$STARTING_CONVERSATION],
  ] as const)("maps %s to its i18n key", (taskStatus, expectedKey) => {
    expect(getTaskStatusI18nKey(taskStatus)).toBe(expectedKey);
  });
});
