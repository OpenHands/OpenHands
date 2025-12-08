import { describe, it, expect } from "vitest";
import { getStatusCode, getIndicatorColor, IndicatorColor } from "#/utils/status";
import { AgentState } from "#/types/agent-state";
import { I18nKey } from "#/i18n/declaration";

describe("getStatusCode", () => {
  it("should prioritize agent readiness over stale runtime status", () => {
    // Test case: Agent is ready (AWAITING_USER_INPUT) but runtime status is still starting
    const result = getStatusCode(
      { id: "", message: "", type: "info", status_update: true }, // statusMessage
      "CONNECTED", // webSocketStatus
      "RUNNING", // conversationStatus
      "STATUS$STARTING_RUNTIME", // runtimeStatus (stale)
      AgentState.AWAITING_USER_INPUT, // agentState (ready)
    );

    // Should return agent state message, not runtime status
    expect(result).toBe(I18nKey.AGENT_STATUS$WAITING_FOR_TASK);
  });

  it("should show runtime status when agent is not ready", () => {
    // Test case: Agent is loading and runtime is starting
    const result = getStatusCode(
      { id: "", message: "", type: "info", status_update: true }, // statusMessage
      "CONNECTED", // webSocketStatus
      "STARTING", // conversationStatus
      "STATUS$STARTING_RUNTIME", // runtimeStatus
      AgentState.LOADING, // agentState (not ready)
    );

    // Should return runtime status since agent is not ready
    expect(result).toBe("STATUS$STARTING_RUNTIME");
  });

  it("should handle agent running state with stale runtime status", () => {
    // Test case: Agent is running but runtime status is stale
    const result = getStatusCode(
      { id: "", message: "", type: "info", status_update: true }, // statusMessage
      "CONNECTED", // webSocketStatus
      "RUNNING", // conversationStatus
      "STATUS$BUILDING_RUNTIME", // runtimeStatus (stale)
      AgentState.RUNNING, // agentState (ready)
    );

    // Should return agent state message, not runtime status
    expect(result).toBe(I18nKey.AGENT_STATUS$RUNNING_TASK);
  });

  it("should handle agent finished state with stale runtime status", () => {
    // Test case: Agent is finished but runtime status is stale
    const result = getStatusCode(
      { id: "", message: "", type: "info", status_update: true }, // statusMessage
      "CONNECTED", // webSocketStatus
      "RUNNING", // conversationStatus
      "STATUS$SETTING_UP_WORKSPACE", // runtimeStatus (stale)
      AgentState.FINISHED, // agentState (ready)
    );

    // Should return agent state message, not runtime status
    expect(result).toBe(I18nKey.AGENT_STATUS$WAITING_FOR_TASK);
  });

  it("should still respect stopped states", () => {
    // Test case: Runtime is stopped - should always show stopped
    const result = getStatusCode(
      { id: "", message: "", type: "info", status_update: true }, // statusMessage
      "CONNECTED", // webSocketStatus
      "STOPPED", // conversationStatus
      "STATUS$STOPPED", // runtimeStatus
      AgentState.RUNNING, // agentState
    );

    // Should return stopped status regardless of agent state
    expect(result).toBe(I18nKey.CHAT_INTERFACE$STOPPED);
  });

  it("should handle null agent state with runtime status", () => {
    // Test case: No agent state, runtime is starting
    const result = getStatusCode(
      { id: "", message: "", type: "info", status_update: true }, // statusMessage
      "CONNECTED", // webSocketStatus
      "STARTING", // conversationStatus
      "STATUS$STARTING_RUNTIME", // runtimeStatus
      null, // agentState
    );

    // Should return runtime status since no agent state
    expect(result).toBe("STATUS$STARTING_RUNTIME");
  });

  it("should prioritize task ERROR status over websocket CONNECTING state", () => {
    // Test case: Task has errored but websocket is still trying to connect
    const result = getStatusCode(
      { id: "", message: "", type: "info", status_update: true }, // statusMessage
      "CONNECTING", // webSocketStatus (stuck connecting)
      null, // conversationStatus
      null, // runtimeStatus
      AgentState.LOADING, // agentState
      "ERROR", // taskStatus (ERROR)
    );

    // Should return error message, not "Connecting..."
    expect(result).toBe(I18nKey.AGENT_STATUS$ERROR_OCCURRED);
  });

  it("should show Connecting when task is working and websocket is connecting", () => {
    // Test case: Task is in progress and websocket is connecting normally
    const result = getStatusCode(
      { id: "", message: "", type: "info", status_update: true }, // statusMessage
      "CONNECTING", // webSocketStatus
      null, // conversationStatus
      null, // runtimeStatus
      AgentState.LOADING, // agentState
      "WORKING", // taskStatus (in progress)
    );

    // Should show connecting message since task hasn't errored
    expect(result).toBe(I18nKey.CHAT_INTERFACE$CONNECTING);
  });
});

describe("getIndicatorColor", () => {
  it("should prioritize agent readiness over stale runtime status for AWAITING_USER_INPUT", () => {
    // Test case: Agent is ready (AWAITING_USER_INPUT) but runtime status is still starting
    const result = getIndicatorColor(
      "CONNECTED", // webSocketStatus
      "RUNNING", // conversationStatus
      "STATUS$STARTING_RUNTIME", // runtimeStatus (stale)
      AgentState.AWAITING_USER_INPUT, // agentState (ready)
    );

    // Should return blue for AWAITING_USER_INPUT, not yellow for stale runtime
    expect(result).toBe(IndicatorColor.BLUE);
  });

  it("should prioritize agent readiness over stale runtime status for RUNNING", () => {
    // Test case: Agent is running but runtime status is stale
    const result = getIndicatorColor(
      "CONNECTED", // webSocketStatus
      "RUNNING", // conversationStatus
      "STATUS$BUILDING_RUNTIME", // runtimeStatus (stale)
      AgentState.RUNNING, // agentState (ready)
    );

    // Should return green for RUNNING, not yellow for stale runtime
    expect(result).toBe(IndicatorColor.GREEN);
  });

  it("should prioritize agent readiness over stale runtime status for FINISHED", () => {
    // Test case: Agent is finished but runtime status is stale
    const result = getIndicatorColor(
      "CONNECTED", // webSocketStatus
      "RUNNING", // conversationStatus
      "STATUS$SETTING_UP_WORKSPACE", // runtimeStatus (stale)
      AgentState.FINISHED, // agentState (ready)
    );

    // Should return green for FINISHED, not yellow for stale runtime
    expect(result).toBe(IndicatorColor.GREEN);
  });

  it("should show yellow when agent is not ready and runtime is starting", () => {
    // Test case: Agent is loading and runtime is starting
    const result = getIndicatorColor(
      "CONNECTED", // webSocketStatus
      "STARTING", // conversationStatus
      "STATUS$STARTING_RUNTIME", // runtimeStatus
      AgentState.LOADING, // agentState (not ready)
    );

    // Should return yellow since agent is not ready
    expect(result).toBe(IndicatorColor.YELLOW);
  });

  it("should show orange for AWAITING_USER_CONFIRMATION even with stale runtime", () => {
    // Test case: Agent is awaiting confirmation but runtime status is stale
    const result = getIndicatorColor(
      "CONNECTED", // webSocketStatus
      "RUNNING", // conversationStatus
      "STATUS$STARTING_RUNTIME", // runtimeStatus (stale)
      AgentState.AWAITING_USER_CONFIRMATION, // agentState (ready)
    );

    // Should return orange for AWAITING_USER_CONFIRMATION, not yellow for stale runtime
    expect(result).toBe(IndicatorColor.ORANGE);
  });

  it("should still respect stopped states", () => {
    // Test case: Runtime is stopped - should always show red
    const result = getIndicatorColor(
      "CONNECTED", // webSocketStatus
      "STOPPED", // conversationStatus
      "STATUS$STOPPED", // runtimeStatus
      AgentState.RUNNING, // agentState
    );

    // Should return red for stopped status regardless of agent state
    expect(result).toBe(IndicatorColor.RED);
  });

  it("should handle null agent state with runtime status", () => {
    // Test case: No agent state, runtime is starting
    const result = getIndicatorColor(
      "CONNECTED", // webSocketStatus
      "STARTING", // conversationStatus
      "STATUS$STARTING_RUNTIME", // runtimeStatus
      null, // agentState
    );

    // Should return yellow since no agent state and runtime is starting
    expect(result).toBe(IndicatorColor.YELLOW);
  });

  it("should handle USER_CONFIRMED state with stale runtime status", () => {
    // Test case: Agent is in USER_CONFIRMED state but runtime status is stale
    const result = getIndicatorColor(
      "CONNECTED", // webSocketStatus
      "RUNNING", // conversationStatus
      "STATUS$BUILDING_RUNTIME", // runtimeStatus (stale)
      AgentState.USER_CONFIRMED, // agentState (ready)
    );

    // Should return green for USER_CONFIRMED, not yellow for stale runtime
    expect(result).toBe(IndicatorColor.GREEN);
  });

  it("should handle USER_REJECTED state with stale runtime status", () => {
    // Test case: Agent is in USER_REJECTED state but runtime status is stale
    const result = getIndicatorColor(
      "CONNECTED", // webSocketStatus
      "RUNNING", // conversationStatus
      "STATUS$BUILDING_RUNTIME", // runtimeStatus (stale)
      AgentState.USER_REJECTED, // agentState (ready)
    );

    // Should return green for USER_REJECTED, not yellow for stale runtime
    expect(result).toBe(IndicatorColor.GREEN);
  });
});
