import ActionType from "#/types/action-type";
import { AgentState } from "#/types/agent-state";

export const generateAgentStateChangeEvent = (state: AgentState) => ({
  action: ActionType.CHANGE_AGENT_STATE,
  args: { agent_state: state },
});
