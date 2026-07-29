import { create } from "zustand";

export enum ActionSecurityRisk {
  UNKNOWN = -1,
  LOW = 0,
  MEDIUM = 1,
  HIGH = 2,
}

export type SecurityAnalyzerLog = {
  id: number;
  content: string;
  security_risk: ActionSecurityRisk;
  confirmation_state?: "awaiting_confirmation" | "confirmed" | "rejected";
  confirmed_changed: boolean;
};

interface SecurityAnalyzerState {
  logs: SecurityAnalyzerLog[];
}

interface SecurityAnalyzerStore extends SecurityAnalyzerState {
  appendSecurityAnalyzerInput: (message: {
    id: number;
    args: {
      command?: string;
      code?: string;
      content?: string;
      security_risk: ActionSecurityRisk;
      confirmation_state?: "awaiting_confirmation" | "confirmed" | "rejected";
    };
    message?: string;
  }) => void;
  clearLogs: () => void;
}

const initialLogs: SecurityAnalyzerLog[] = [];

export const useSecurityAnalyzerStore = create<SecurityAnalyzerStore>(
  (set) => ({
    logs: initialLogs,
    appendSecurityAnalyzerInput: (message) =>
      set((state) => {
        const log: SecurityAnalyzerLog = {
          id: message.id,
          content:
            message.args.command ||
            message.args.code ||
            message.args.content ||
            message.message ||
            "",
          security_risk: message.args.security_risk,
          confirmation_state: message.args.confirmation_state,
          confirmed_changed: false,
        };

        const existingLog = state.logs.find(
          (stateLog) =>
            stateLog.id === log.id ||
            (stateLog.confirmation_state === "awaiting_confirmation" &&
              stateLog.content === log.content),
        );

        if (existingLog) {
          if (existingLog.confirmation_state !== log.confirmation_state) {
            existingLog.confirmation_state = log.confirmation_state;
            existingLog.confirmed_changed = true;
          }
          return { logs: [...state.logs] }; // Return new array to trigger re-render
        }
        return { logs: [...state.logs, log] };
      }),
    clearLogs: () => set({ logs: initialLogs }),
  }),
);
