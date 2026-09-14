import React, {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
} from "react";
import type {
  ConversationOverviewDrawerOpenOptions,
  ConversationOverviewDrawerSection,
} from "./conversation-overview-drawer.types";

interface ConversationOverviewDrawerContextValue {
  section: ConversationOverviewDrawerSection | null;
  openAdd: boolean;
  /** Increments when the drawer header Add control is clicked. */
  addRequestKey: number;
  openSection: (
    section: ConversationOverviewDrawerSection,
    options?: ConversationOverviewDrawerOpenOptions,
  ) => void;
  closeDrawer: () => void;
  requestAdd: () => void;
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
  const [openAdd, setOpenAdd] = useState(false);
  const [addRequestKey, setAddRequestKey] = useState(0);

  const openSection = useCallback(
    (
      nextSection: ConversationOverviewDrawerSection,
      options?: ConversationOverviewDrawerOpenOptions,
    ) => {
      setSection(nextSection);
      setOpenAdd(Boolean(options?.openAdd));
      setAddRequestKey(0);
    },
    [],
  );

  const closeDrawer = useCallback(() => {
    setSection(null);
    setOpenAdd(false);
    setAddRequestKey(0);
  }, []);

  const requestAdd = useCallback(() => {
    setAddRequestKey((previous) => previous + 1);
  }, []);

  const value = useMemo(
    () => ({
      section,
      openAdd,
      addRequestKey,
      openSection,
      closeDrawer,
      requestAdd,
    }),
    [addRequestKey, closeDrawer, openAdd, openSection, requestAdd, section],
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
