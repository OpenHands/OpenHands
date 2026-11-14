import React, {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
} from "react";
import type { ConversationOverviewDrawerSection } from "./conversation-overview-drawer.types";

interface ConversationOverviewDrawerContextValue {
  section: ConversationOverviewDrawerSection | null;
  openSection: (section: ConversationOverviewDrawerSection) => void;
  closeDrawer: () => void;
}

const ConversationOverviewDrawerContext =
  createContext<ConversationOverviewDrawerContextValue | null>(null);

export function ConversationOverviewDrawerProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const [section, setSection] =
    useState<ConversationOverviewDrawerSection | null>(null);

  const closeDrawer = useCallback(() => setSection(null), []);

  const value = useMemo(
    () => ({ section, openSection: setSection, closeDrawer }),
    [closeDrawer, section],
  );

  return (
    <ConversationOverviewDrawerContext.Provider value={value}>
      {children}
    </ConversationOverviewDrawerContext.Provider>
  );
}

export function useConversationOverviewDrawer() {
  const context = useContext(ConversationOverviewDrawerContext);
  if (!context) {
    throw new Error(
      "useConversationOverviewDrawer must be used within ConversationOverviewDrawerProvider",
    );
  }
  return context;
}

export function useConversationOverviewDrawerOptional() {
  return useContext(ConversationOverviewDrawerContext);
}
