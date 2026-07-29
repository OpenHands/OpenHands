import { create } from "zustand";
import { AgentState } from "#/types/agent-state";

interface AgentStateData {
  curAgentState: AgentState;
}

interface AgentStore extends AgentStateData {
  setCurrentAgentState: (state: AgentState) => void;
  reset: () => void;
}

const initialState: AgentStateData = {
  curAgentState: AgentState.LOADING,
};

export const useAgentStore = create<AgentStore>((set) => ({
  ...initialState,
  setCurrentAgentState: (state: AgentState) => set({ curAgentState: state }),
  reset: () => set(initialState),
}));
