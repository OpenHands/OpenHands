import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import type { ScaScanResult, SecurityScanResult } from "#/types/security-scan";

export const SECURITY_SCAN_RESULTS_STORAGE_KEY =
  "openhands-security-scan-results";

export interface ConversationSecurityScanResults {
  sast: SecurityScanResult | null;
  sca: ScaScanResult | null;
}

interface SecurityScanResultsState {
  resultsByConversationId: Record<string, ConversationSecurityScanResults>;
}

interface SecurityScanResultsActions {
  setSastResult: (
    conversationId: string,
    result: SecurityScanResult | null,
  ) => void;
  setScaResult: (conversationId: string, result: ScaScanResult | null) => void;
  clearConversation: (conversationId: string) => void;
}

export type SecurityScanResultsStore = SecurityScanResultsState &
  SecurityScanResultsActions;

const EMPTY_RESULTS: ConversationSecurityScanResults = {
  sast: null,
  sca: null,
};

function getResultsForConversation(
  resultsByConversationId: Record<string, ConversationSecurityScanResults>,
  conversationId: string,
): ConversationSecurityScanResults {
  return resultsByConversationId[conversationId] ?? EMPTY_RESULTS;
}

export const useSecurityScanResultsStore = create<SecurityScanResultsStore>()(
  persist(
    (set, get) => ({
      resultsByConversationId: {},

      setSastResult: (conversationId, result) => {
        const current = getResultsForConversation(
          get().resultsByConversationId,
          conversationId,
        );
        set((state) => ({
          resultsByConversationId: {
            ...state.resultsByConversationId,
            [conversationId]: {
              ...current,
              sast: result,
            },
          },
        }));
      },

      setScaResult: (conversationId, result) => {
        const current = getResultsForConversation(
          get().resultsByConversationId,
          conversationId,
        );
        set((state) => ({
          resultsByConversationId: {
            ...state.resultsByConversationId,
            [conversationId]: {
              ...current,
              sca: result,
            },
          },
        }));
      },

      clearConversation: (conversationId) => {
        if (!(conversationId in get().resultsByConversationId)) {
          return;
        }
        set((state) => {
          const next = { ...state.resultsByConversationId };
          delete next[conversationId];
          return { resultsByConversationId: next };
        });
      },
    }),
    {
      name: SECURITY_SCAN_RESULTS_STORAGE_KEY,
      storage: createJSONStorage(() => localStorage),
      partialize: (state): SecurityScanResultsState => ({
        resultsByConversationId: state.resultsByConversationId,
      }),
    },
  ),
);
